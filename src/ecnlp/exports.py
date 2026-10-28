"""Analysis-ready exports and the coverage report (SDS: sample attrition)."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def build_ai_contexts(con: sqlite3.Connection, window: int = 1) -> pd.DataFrame:
    """Each AI-term sentence with +/- `window` sentences from the same speaker turn.

    This is the unit the SDS specifies for FinBERT and BERTopic (mention + context).
    """
    s = db.read_df(con, "SELECT sentence_id, call_id, segment_id, sent_idx, section, speaker_type, text, "
                        "is_boilerplate, has_ai_term, ai_terms FROM sentences ORDER BY call_id, segment_id, sent_idx")
    if s.empty:
        return s
    rows = []
    for seg_id, g in s.groupby("segment_id", sort=False):
        texts = g["text"].tolist()
        for k, r in enumerate(g.itertuples(index=False)):
            if not r.has_ai_term or r.is_boilerplate:
                continue
            lo, hi = max(0, k - window), min(len(texts), k + window + 1)
            rows.append({"sentence_id": r.sentence_id, "call_id": r.call_id, "segment_id": seg_id,
                         "section": r.section, "speaker_type": r.speaker_type, "ai_terms": r.ai_terms,
                         "sentence": r.text, "context_text": " ".join(texts[lo:hi])})
    return pd.DataFrame(rows)


def export_all(cfg: Config, con: sqlite3.Connection) -> dict[str, str]:
    out = cfg.paths.processed
    files: dict[str, str] = {}

    meta_cols = ("c.ticker, c.call_key, c.fiscal_year, c.fiscal_quarter, c.call_date_et, c.call_session, "
                 "co.sector, co.yf_sector, co.industry")
    sent = db.read_df(con, f"""
        SELECT s.*, {meta_cols}
        FROM sentences s JOIN calls c USING(call_id) LEFT JOIN companies co ON co.ticker = c.ticker
        WHERE s.is_boilerplate = 0
        ORDER BY s.call_id, s.segment_id, s.sent_idx""")
    files["sentences"] = str(_write(sent, out / "nlp_sentences.parquet"))

    ctx = build_ai_contexts(con)
    if not ctx.empty:
        meta = db.read_df(con, f"SELECT c.call_id, {meta_cols} FROM calls c LEFT JOIN companies co ON co.ticker=c.ticker")
        ctx = ctx.merge(meta, on="call_id", how="left")
    files["ai_contexts"] = str(_write(ctx, out / "nlp_ai_contexts.parquet"))

    segs = db.read_df(con, f"""
        SELECT g.segment_id, g.call_id, g.component_order, g.speaker_name, g.speaker_type, g.speaker_title,
               g.section, g.text_clean, g.n_words, {meta_cols}
        FROM segments g JOIN calls c USING(call_id) LEFT JOIN companies co ON co.ticker = c.ticker
        WHERE g.is_operator = 0 ORDER BY g.call_id, g.component_order""")
    files["segments"] = str(_write(segs, out / "nlp_segments.parquet"))

    # One row per call: transcript stats + company attributes + market reaction.
    has_er = con.execute("SELECT name FROM sqlite_master WHERE name='event_returns'").fetchone()
    er_join = "LEFT JOIN event_returns e ON e.call_id = c.call_id" if has_er else ""
    er_cols = ", e.*" if has_er else ""
    master = db.read_df(con, f"""
        SELECT c.*, co.company_name AS co_name, co.sector, co.industry, co.yf_sector, co.yf_industry,
               co.yf_market_cap {er_cols}
        FROM calls c LEFT JOIN companies co ON co.ticker = c.ticker {er_join}
        ORDER BY c.ticker, c.call_date_et""")
    master = master.loc[:, ~master.columns.duplicated()]
    if "n_words_ex_operator" in master and "n_ai_sentences" in master:
        master["ai_sentences_per_1000_words"] = master["n_ai_sentences"] / master["n_words_ex_operator"].where(
            master["n_words_ex_operator"] > 0) * 1000
    files["calls_master"] = str(_write(master, out / "calls_master.csv"))

    cov = coverage_report(con)
    files["coverage"] = str(_write(cov, out / "coverage_report.csv"))
    for k, v in files.items():
        log.info("export %-12s -> %s", k, v)
    return files


def coverage_report(con: sqlite3.Connection) -> pd.DataFrame:
    has_er = con.execute("SELECT name FROM sqlite_master WHERE name='event_returns'").fetchone()
    er = db.read_df(con, "SELECT * FROM event_returns") if has_er else pd.DataFrame()
    calls = db.read_df(con, "SELECT ticker, call_id, status, n_ai_sentences FROM calls")
    cos = db.read_df(con, "SELECT ticker, company_name, n_calls_api, n_calls_ai, yf_symbol, yf_status FROM companies")
    rows = []
    for t in sorted(set(cos["ticker"]) | set(calls["ticker"])):
        c = calls[calls["ticker"] == t]
        co = cos[cos["ticker"] == t]
        e = er[er["ticker"] == t] if len(er) else pd.DataFrame()
        cars = e.filter(like="car_mkt_") if len(e) else pd.DataFrame()
        rows.append({
            "ticker": t,
            "company_name": co["company_name"].iloc[0] if len(co) else None,
            "calls_in_api": co["n_calls_api"].iloc[0] if len(co) else None,
            "calls_mentioning_ai": co["n_calls_ai"].iloc[0] if len(co) else None,
            "calls_selected": len(c),
            "calls_with_text": int((c["status"] == "ok").sum()),
            "calls_no_text": int((c["status"] == "no_text").sum()),
            "calls_error": int((c["status"] == "error").sum()),
            "calls_pending": int((c["status"] == "pending").sum()),
            "calls_with_ai_keyword": int((c["n_ai_sentences"].fillna(0) > 0).sum()),
            "yf_status": co["yf_status"].iloc[0] if len(co) else None,
            "events_all_windows": int(cars.notna().all(axis=1).sum()) if len(cars) and cars.shape[1] else 0,
        })
    return pd.DataFrame(rows)
