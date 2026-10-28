"""NLP-01: pull earnings call transcripts (speaker segments) from earningscalls.dev.

Request plan (keeps a 500-ticker universe well inside the Ultra 20k/month quota):

1. Discovery  - /search/by_ticker with the AI query: ONE request per 100
   tickers returns the ids of every call that mentions AI.
2. Call lists - /earnings/?ticker=X&country=US&event_type=earnings: one request
   per ticker (per 100 calls) for dates, fiscal period and company metadata.
   Intersecting with step 1 drops non-earnings events and foreign listings
   that share a symbol.
3. Text       - /speakers/{id}: usually one request per call (200 turns/page).
   Falls back to /transcripts/{id} for calls without speaker segments.

Everything is cached on disk; transcripts are immutable so a second run makes
no transcript requests at all.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from . import db
from .cleaning import assign_sections, normalize_text, parse_fiscal_period, word_count
from .config import Config
from .earningscalls_client import EarningsCallsClient, QuotaExceeded
from .timeutils import classify_session, parse_event_time

log = logging.getLogger(__name__)

SEGMENT_PAGE_SIZE = 200          # API maximum for /speakers
MAX_PAGES = 50                   # safety stop for pagination loops


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def base_ticker(t: Any) -> str:
    """'AAPL:US' -> 'AAPL'. Dots are kept (BRK.B is a real US symbol)."""
    return str(t or "").split(":")[0].strip().upper()


def _chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _short_hash(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------- discovery
def discover_ai_calls(client: EarningsCallsClient, cfg: Config) -> dict[str, set[int]]:
    """ticker -> set of earnings call ids whose transcript matches the AI query."""
    q = cfg.ec.get("ai_query")
    ttl = cfg.ec.get("call_list_cache_days", 1)
    out: dict[str, set[int]] = {t: set() for t in cfg.tickers}
    needs_fallback: list[str] = []

    for chunk in _chunks(cfg.tickers, 100):
        params = {"q": q, "tickers": ",".join(chunk), "date_from": cfg.date_from,
                  "date_to": cfg.date_to, "limit": 500}
        payload = client.get("search/by_ticker", params=params,
                             cache_key=f"ai_by_ticker_{_short_hash(params)}", max_age_days=ttl) or {}
        results = payload.get("results") or payload.get("data") or []
        log.info("AI discovery: %s of %s tickers matched (%s calls total)",
                 payload.get("total_companies", len(results)), len(chunk), payload.get("total_calls", "?"))
        for row in results:
            t = base_ticker(row.get("ticker"))
            if t not in out:
                continue  # an alias we did not ask for
            ids = {int(i) for i in (row.get("earnings_ids") or []) if str(i).isdigit()}
            out[t] |= ids
            matched = row.get("calls_matched")
            if isinstance(matched, int) and matched > len(ids):
                needs_fallback.append(t)

    # If the aggregate endpoint truncated the id list, page through /search for that ticker.
    for t in needs_fallback:
        log.info("%s: by_ticker returned a truncated id list; using /search fallback", t)
        out[t] |= _search_ids_for_ticker(client, cfg, t)
    return out


def _search_ids_for_ticker(client: EarningsCallsClient, cfg: Config, ticker: str) -> set[int]:
    ids: set[int] = set()
    ttl = cfg.ec.get("call_list_cache_days", 1)
    for page in range(1, MAX_PAGES + 1):
        params = {"q": cfg.ec.get("ai_query"), "type": "transcripts", "ticker": ticker,
                  "date_from": cfg.date_from, "date_to": cfg.date_to, "limit": 100, "page": page}
        payload = client.get("search/", params=params,
                             cache_key=f"ai_search_{ticker}_{_short_hash(params)}", max_age_days=ttl) or {}
        items = payload.get("data") or payload.get("results") or []
        if isinstance(items, dict):  # type=all style response
            items = items.get("transcripts") or []
        for it in items:
            cid = it.get("earnings_call_id") or it.get("earnings_id") or it.get("id")
            if str(cid).isdigit():
                ids.add(int(cid))
        if len(items) < 100:
            break
    return ids


# --------------------------------------------------------------- call lists
def list_ticker_calls(client: EarningsCallsClient, cfg: Config, ticker: str) -> tuple[list[dict], dict]:
    """All earnings calls for a ticker in the date range, plus response metadata."""
    ttl = cfg.ec.get("call_list_cache_days", 1)
    rows: list[dict] = []
    meta: dict = {}
    for page in range(1, MAX_PAGES + 1):
        params = {"ticker": ticker, "country": cfg.ec.get("country"), "event_type": cfg.ec.get("event_type"),
                  "date_from": cfg.date_from, "date_to": cfg.date_to, "sort": "date_asc",
                  "limit": 100, "page": page}
        payload = client.get("earnings/", params=params,
                             cache_key=f"calls_{ticker}_{_short_hash(params)}", max_age_days=ttl) or {}
        if page == 1:
            meta = {k: payload.get(k) for k in ("ambiguous", "issuer_count")}
        data = payload.get("data") or []
        rows.extend(data)
        pag = payload.get("pagination") or {}
        total_pages = pag.get("total_pages")
        if (total_pages is not None and page >= total_pages) or len(data) < 100:
            break
    return rows, meta


def _company_row(ticker: str, rows: list[dict], meta: dict) -> tuple[dict, list[dict]]:
    """Pick the issuer for this ticker and return (company row, rows for that issuer)."""
    if rows:
        issuers = Counter(r.get("issuer_id") for r in rows if r.get("issuer_id"))
        if len(issuers) > 1:
            keep, n = issuers.most_common(1)[0]
            log.warning("%s maps to %d issuers %s; keeping the most frequent (%s, %d calls)",
                        ticker, len(issuers), dict(issuers), keep, n)
            rows = [r for r in rows if r.get("issuer_id") in (keep, None)]
    first = rows[-1] if rows else {}   # newest row has the most current metadata
    return ({
        "ticker": ticker,
        "company_name": first.get("company_name"),
        "sector": first.get("sector"),
        "industry": first.get("industry"),
        "exchange": first.get("exchange"),
        "mic": first.get("mic"),
        "country": first.get("country"),
        "currency": first.get("currency"),
        "issuer_id": first.get("issuer_id"),
        "isin": first.get("isin"),
        "ec_ambiguous": int(bool(meta.get("ambiguous"))) if meta.get("ambiguous") is not None else None,
        "ec_issuer_count": meta.get("issuer_count"),
        "updated_at": _now(),
    }, rows)


def _call_row(ticker: str, r: dict, mentions_ai: bool, cfg: Config) -> dict:
    fy, fq = r.get("fiscal_year"), r.get("fiscal_quarter")
    if not fy or not fq:
        pfy, pfq = parse_fiscal_period(r.get("transcript_title"))
        fy, fq = fy or pfy, fq or pfq
    ts, known = parse_event_time(r.get("event_date_time"), cfg.ec.get("naive_timestamp_tz", "UTC"))
    timing = classify_session(ts, known, r.get("event_date_time"),
                              cfg.events.get("market_open", "09:30"), cfg.events.get("market_close", "16:00"))
    return {
        "call_id": int(r["id"]),
        "call_key": f"{ticker}_{fy}Q{fq}" if fy and fq else None,
        "ticker": ticker,
        "company_name": r.get("company_name"),
        "transcript_title": r.get("transcript_title"),
        "event_type": r.get("event_type"),
        "fiscal_year": fy,
        "fiscal_quarter": fq,
        "event_datetime_raw": r.get("event_date_time"),
        **timing,
        "mentions_ai": int(mentions_ai),
        "ai_query": cfg.ec.get("ai_query"),
    }


# ---------------------------------------------------------------- segments
def _as_list(payload: Any) -> list[dict]:
    if payload is None:
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        for k in ("segments", "speakers", "components", "data"):
            if isinstance(data.get(k), list):
                return data[k]
        return []
    return data if isinstance(data, list) else []


def _norm_segment(seg: dict, i: int) -> dict:
    text = seg.get("text_content") or seg.get("text") or seg.get("content") or ""
    order = seg.get("component_order", seg.get("order", seg.get("index", i)))
    try:
        order = int(order)
    except (TypeError, ValueError):
        order = i
    return {
        "component_order": order,
        "speaker_name": seg.get("speaker_name") or seg.get("name"),
        "speaker_type": (seg.get("speaker_type") or seg.get("role") or "unknown").lower(),
        "speaker_title": seg.get("speaker_title") or seg.get("title") or seg.get("speaker_role"),
        "text_raw": text,
    }


def fetch_segments(client: EarningsCallsClient, call_id: int) -> list[dict]:
    segs: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        payload = client.speakers(call_id, page=page, limit=SEGMENT_PAGE_SIZE)
        items = _as_list(payload)
        segs.extend(items)
        pag = (payload or {}).get("pagination") or {}
        total_pages = pag.get("total_pages")
        has_more = pag.get("has_more")
        if not items or len(items) < SEGMENT_PAGE_SIZE or has_more is False or (
                total_pages is not None and page >= total_pages):
            break
    out = [_norm_segment(s, i) for i, s in enumerate(segs)]
    out = [s for s in out if s["text_raw"].strip()]
    out.sort(key=lambda s: s["component_order"])
    # de-duplicate orders (defensive against overlapping pages)
    seen, uniq = set(), []
    for s in out:
        if s["component_order"] not in seen:
            seen.add(s["component_order"])
            uniq.append(s)
    return uniq


def _longest_string(obj: Any) -> str:
    best = ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            s = _longest_string(v)
            if len(s) > len(best):
                best = s
    elif isinstance(obj, list):
        for v in obj:
            s = _longest_string(v)
            if len(s) > len(best):
                best = s
    return best


def fetch_transcript_text(client: EarningsCallsClient, call_id: int) -> tuple[str, list[dict]]:
    """Fallback for calls without speaker segments: full text via /transcripts/{id}.

    Returns (text, segments). Some responses embed a `segments` list; if so it
    is used, otherwise the text comes back as a single unlabeled segment.
    """
    parts: list[str] = []
    offset = 0
    for _ in range(MAX_PAGES):
        payload = client.transcript(call_id, offset=offset)
        if not payload:
            break
        embedded = _as_list(payload)
        if embedded and offset == 0 and any(isinstance(s, dict) and (s.get("text_content") or s.get("text")) for s in embedded):
            segs = [_norm_segment(s, i) for i, s in enumerate(embedded)]
            return " ".join(s["text_raw"] for s in segs), segs
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        text = None
        if isinstance(data, dict):
            for k in ("transcript_text", "full_transcript", "transcript", "text", "content", "full_text"):
                if isinstance(data.get(k), str):
                    text = data[k]
                    break
        if text is None:
            text = _longest_string(data)
        parts.append(text or "")
        nxt = (data.get("next_offset") if isinstance(data, dict) else None) or (
            payload.get("next_offset") if isinstance(payload, dict) else None)
        if not nxt or int(nxt) <= offset:
            break
        offset = int(nxt)
    full = "".join(parts)
    segs = [{"component_order": 0, "speaker_name": None, "speaker_type": "unknown",
             "speaker_title": None, "text_raw": full}] if full.strip() else []
    return full, segs


def _segment_rows(call_id: int, segs: list[dict]) -> list[dict]:
    sections = assign_sections([s["speaker_type"] for s in segs])
    rows = []
    for s, sec in zip(segs, sections):
        rows.append({
            "segment_id": f"{call_id}-{s['component_order']}",
            "call_id": call_id,
            "component_order": s["component_order"],
            "speaker_name": s["speaker_name"],
            "speaker_type": s["speaker_type"],
            "speaker_title": s["speaker_title"],
            "section": sec,
            "text_raw": s["text_raw"],
            "is_operator": int(s["speaker_type"] == "operator" or (s["speaker_name"] or "").strip().lower() == "operator"),
            "n_words": word_count(normalize_text(s["text_raw"])),
        })
    return rows


# --------------------------------------------------------------------- run
def pull_transcripts(cfg: Config, client: EarningsCallsClient, con: sqlite3.Connection,
                     remaining_quota: int | None = None) -> dict[str, Any]:
    include_all = bool(cfg.ec.get("include_non_ai_calls", False))
    ai_ids = discover_ai_calls(client, cfg)

    selected: list[dict] = []
    for ticker in cfg.tickers:
        rows, meta = list_ticker_calls(client, cfg, ticker)
        company, rows = _company_row(ticker, rows, meta)
        listed_ids = {int(r["id"]) for r in rows if str(r.get("id")).isdigit()}
        ai_here = ai_ids.get(ticker, set())
        dropped = ai_here - listed_ids
        if dropped:
            log.debug("%s: %d AI-matching events are not US earnings calls in range; ignored", ticker, len(dropped))
        chosen = [r for r in rows if str(r.get("id")).isdigit() and (include_all or int(r["id"]) in ai_here)]
        company["n_calls_api"] = len(rows)
        company["n_calls_ai"] = len(listed_ids & ai_here)
        db.update_partial(con, "companies", "ticker", [company])
        if not rows:
            log.warning("%s: no earnings calls found on earningscalls.dev in %s..%s", ticker, cfg.date_from, cfg.date_to)
        for r in chosen:
            selected.append(_call_row(ticker, r, int(r["id"]) in ai_here, cfg))
        con.commit()

    # Same call reached through two tickers (e.g. share classes) -> keep first.
    uniq: dict[int, dict] = {}
    for c in selected:
        uniq.setdefault(c["call_id"], c)
    if len(uniq) < len(selected):
        log.warning("%d duplicate calls across tickers removed (check for multiple share classes)",
                    len(selected) - len(uniq))
    selected = list(uniq.values())

    # Duplicate fiscal periods for one ticker get a suffix so call_key stays unique.
    key_counts: Counter = Counter()
    for c in sorted(selected, key=lambda c: (c["ticker"], c["event_datetime_raw"] or "")):
        if c["call_key"]:
            key_counts[c["call_key"]] += 1
            if key_counts[c["call_key"]] > 1:
                c["call_key"] = f"{c['call_key']}_{key_counts[c['call_key']]}"

    # Write call metadata, preserving status of calls already pulled.
    done = {r[0] for r in con.execute("SELECT call_id FROM calls WHERE status='ok'")}
    for c in selected:
        if c["call_id"] not in done:
            c["status"] = "pending"
    db.update_partial(con, "calls", "call_id", selected)
    con.commit()

    todo = [c for c in selected if c["call_id"] not in done]
    est = int(len(todo) * 1.2) + 1
    log.info("Calls selected: %d (%d already pulled, %d to fetch, ~%d requests)",
             len(selected), len(selected) - len(todo), len(todo), est)
    if remaining_quota is not None:
        margin = int(cfg.ec.get("monthly_safety_margin", 500))
        budget = remaining_quota - margin
        if est > budget:
            log.warning("Estimated requests (%d) exceed remaining monthly budget (%d after margin). "
                        "Pulling as many as fit; re-run next billing period for the rest.", est, max(budget, 0))
            client.max_requests = min(client.max_requests or budget, client.requests_made + max(budget, 0))

    stats = Counter()
    for n, c in enumerate(todo, 1):
        cid = c["call_id"]
        try:
            segs = fetch_segments(client, cid)
            source = "speakers"
            if not segs:
                _, segs = fetch_transcript_text(client, cid)
                source = "transcript" if segs else "none"
        except QuotaExceeded as exc:
            log.error("Stopping transcript pull: %s", exc)
            stats["stopped_quota"] += 1
            break
        except Exception as exc:  # keep going; record the failure
            log.exception("Call %s failed", cid)
            db.update_partial(con, "calls", "call_id", [{"call_id": cid, "status": "error",
                                                          "status_detail": str(exc)[:500], "pulled_at": _now()}])
            stats["error"] += 1
            continue

        seg_rows = _segment_rows(cid, segs)
        db.replace_where(con, "segments", "call_id=?", (cid,), seg_rows)
        labeled = any(s["speaker_type"] not in ("unknown", "", None) for s in segs)
        db.update_partial(con, "calls", "call_id", [{
            "call_id": cid,
            "text_source": source,
            "has_speaker_labels": int(labeled),
            "n_segments": len(seg_rows),
            "n_words": sum(r["n_words"] for r in seg_rows),
            "n_words_ex_operator": sum(r["n_words"] for r in seg_rows if not r["is_operator"]),
            "status": "ok" if seg_rows else "no_text",
            "status_detail": None if seg_rows else "API returned no transcript text",
            "pulled_at": _now(),
        }])
        stats["ok" if seg_rows else "no_text"] += 1
        if n % 25 == 0:
            con.commit()
            log.info("  %d/%d calls pulled (%d API requests so far)", n, len(todo), client.requests_made)
    con.commit()
    stats["selected"] = len(selected)
    log.info("Transcript pull finished: %s", dict(stats))
    return dict(stats)
