"""Command-line entry point.

    python -m ecnlp.pipeline check         # verify API key, quota and live response schema (~4 requests)
    python -m ecnlp.pipeline transcripts   # NLP-01
    python -m ecnlp.pipeline market        # NLP-02
    python -m ecnlp.pipeline prepare       # clean + sentence-split (offline)
    python -m ecnlp.pipeline events        # event-window returns (offline)
    python -m ecnlp.pipeline export        # parquet/csv exports + coverage report (offline)
    python -m ecnlp.pipeline all           # everything above except check

Options: --config PATH, --tickers NVDA,MSFT (override the list), --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone

from . import db
from .config import Config, ConfigError, load_config
from .earningscalls_client import EarningsCallsClient, EarningsCallsError, parse_usage
from .logging_utils import setup_logging

log = logging.getLogger("ecnlp")

STAGES = ["transcripts", "market", "prepare", "events", "export"]


def make_client(cfg: Config) -> EarningsCallsClient:
    return EarningsCallsClient(
        api_key=cfg.api_key or "",
        provider=cfg.ec.get("provider", "direct"),
        cache_dir=cfg.paths.raw_ec,
        requests_per_minute=int(cfg.ec.get("requests_per_minute", 55)),
        max_requests=int(cfg.ec.get("max_requests_per_run", 6000)),
    )


def remaining_quota(client: EarningsCallsClient, cfg: Config) -> int | None:
    try:
        me = client.me()
    except EarningsCallsError as exc:
        log.warning("Could not read usage from /me (%s); relying on max_requests_per_run", exc)
        return None
    usage = parse_usage(me)
    log.info("API usage this period: used=%s limit=%s remaining=%s", usage["used"], usage["limit"], usage["remaining"])
    if usage["remaining"] is None and usage["used"] is not None:
        return int(cfg.ec.get("monthly_limit", 20000)) - usage["used"]
    return usage["remaining"]


def stage_check(cfg: Config) -> int:
    """Smoke test against the live API. Prints what the code relies on."""
    client = make_client(cfg)
    print("\n== /me ==")
    me = client.me()
    print(json.dumps(me, indent=2)[:1500])
    print("parsed usage:", parse_usage(me))

    t = cfg.tickers[0]
    print(f"\n== /earnings/?ticker={t} (newest 3) ==")
    lst = client.list_earnings(ticker=t, country=cfg.ec.get("country"), event_type="earnings",
                               sort="date_desc", limit=3) or {}
    data = lst.get("data") or []
    if not data:
        print("No calls returned - check the ticker / country filter.")
        return 1
    print("item keys:", sorted(data[0].keys()))
    for d in data:
        print(f"  id={d.get('id')} event_date_time={d.get('event_date_time')!r} tz={d.get('timezone')} "
              f"FY{d.get('fiscal_year')}Q{d.get('fiscal_quarter')}  {d.get('transcript_title')}")
    print("  -> US calls are usually ~16:30-17:00 ET (after close) or ~08:00-08:30 ET (pre-market).")
    print("     If these times look 4-5 hours off, set earningscalls.naive_timestamp_tz in config.yaml.")

    cid = data[0]["id"]
    print(f"\n== /speakers/{cid}?limit=5 ==")
    sp = client.get(f"speakers/{cid}", params={"limit": 5}) or {}
    print("top-level keys:", sorted(sp.keys()) if isinstance(sp, dict) else type(sp))
    items = sp.get("data") if isinstance(sp, dict) else None
    if isinstance(items, list) and items:
        print("segment keys:", sorted(items[0].keys()))
        for s in items[:3]:
            print(f"  [{s.get('component_order')}] {s.get('speaker_type')} / {s.get('speaker_name')}: "
                  f"{str(s.get('text_content'))[:90]!r}")
    print("pagination:", sp.get("pagination") if isinstance(sp, dict) else None)

    print("\n== /search/by_ticker (AI query, pilot tickers, last 12 months) ==")
    agg = client.search_by_ticker(q=cfg.ec.get("ai_query"), tickers=",".join(cfg.tickers),
                                  date_from=str(int(cfg.date_to[:4]) - 1) + cfg.date_to[4:], date_to=cfg.date_to) or {}
    print({k: agg.get(k) for k in ("tickers_scoped", "total_companies", "total_calls", "results_count")})
    for r in (agg.get("results") or [])[:5]:
        print(f"  {r.get('ticker')}: calls_matched={r.get('calls_matched')} ids={len(r.get('earnings_ids') or [])}")
    print(f"\nRequests used by this check: {client.requests_made}")
    return 0


def run(stages: list[str], cfg: Config) -> int:
    cfg.paths.ensure()
    con = db.connect(cfg.paths.db)
    run_id = uuid.uuid4().hex[:8]
    rc = 0
    for stage in stages:
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        api_req = cache_hits = 0
        status, notes = "ok", ""
        log.info("===== stage: %s =====", stage)
        try:
            if stage == "transcripts":
                from .transcripts import pull_transcripts

                client = make_client(cfg)
                rem = remaining_quota(client, cfg)
                res = pull_transcripts(cfg, client, con, remaining_quota=rem)
                api_req, cache_hits = client.requests_made, client.cache_hits
                notes = json.dumps(res)
                if res.get("stopped_quota"):
                    status, rc = "partial", 2
            elif stage == "market":
                from .market_data import pull_market_data

                res = pull_market_data(cfg, con)
                notes = json.dumps(res)
                if res["failed"]:
                    status = "partial"
            elif stage == "prepare":
                from .prepare import prepare_text

                notes = json.dumps(prepare_text(cfg, con))
            elif stage == "events":
                from .events import compute_event_returns

                df = compute_event_returns(cfg, con)
                notes = f"{len(df)} calls"
            elif stage == "export":
                from .exports import export_all

                notes = json.dumps(export_all(cfg, con))
        except EarningsCallsError as exc:
            log.error("%s stage stopped: %s", stage, exc)
            status, notes, rc = "error", str(exc), 1
        except Exception as exc:
            log.exception("%s stage failed", stage)
            status, notes, rc = "error", repr(exc), 1
        db.upsert(con, "run_log", [{
            "run_id": run_id, "stage": stage, "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "api_requests": api_req, "cache_hits": cache_hits, "status": status, "notes": notes[:2000]}])
        con.commit()
        if status == "error" and stage == "transcripts" and "401" in notes:
            break  # nothing downstream will work without transcripts metadata
    con.close()
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Earnings call NLP data pipeline")
    ap.add_argument("stage", choices=["check", "all", *STAGES])
    ap.add_argument("--config", default=None)
    ap.add_argument("--tickers", default=None, help="comma-separated override of the ticker list")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    try:
        overrides = {"tickers": args.tickers.split(","), "tickers_csv": None} if args.tickers else None
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    cfg.paths.ensure()
    log_file = setup_logging(cfg.paths.logs, args.stage, logging.DEBUG if args.verbose else logging.INFO)
    log.info("Config: %d tickers, %s..%s, data_dir=%s, log=%s",
             len(cfg.tickers), cfg.date_from, cfg.date_to, cfg.paths.root, log_file)
    if args.stage == "check":
        try:
            return stage_check(cfg)
        except EarningsCallsError as exc:
            log.error("Check failed: %s", exc)
            return 1
    return run(STAGES if args.stage == "all" else [args.stage], cfg)


if __name__ == "__main__":
    sys.exit(main())
