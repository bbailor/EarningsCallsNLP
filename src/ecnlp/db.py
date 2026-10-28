"""SQLite storage (SDS Section 5).

Tables created here are the ingestion/preparation half of the SDS design:
companies, calls, segments, sentences, market_data, yf_earnings_dates,
run_log, plus event_returns (rebuilt by events.py). The NLP tables (tech_mentions, topics, annotations,
call_scores) are created later by the NLP stage and join on call_id /
sentence_id.

`segments` is new relative to the SDS ERD: it holds one row per speaker turn
so sentences can be traced to a speaker and section.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    ticker            TEXT PRIMARY KEY,
    company_name      TEXT,
    sector            TEXT,
    industry          TEXT,
    exchange          TEXT,
    mic               TEXT,
    country           TEXT,
    currency          TEXT,
    issuer_id         TEXT,
    isin              TEXT,
    ec_ambiguous      INTEGER,
    ec_issuer_count   INTEGER,
    n_calls_api       INTEGER,
    n_calls_ai        INTEGER,
    yf_symbol         TEXT,
    yf_sector         TEXT,
    yf_industry       TEXT,
    yf_market_cap     REAL,
    yf_shares_out     REAL,
    yf_quote_type     TEXT,
    yf_status         TEXT,
    updated_at        TEXT
);

CREATE TABLE IF NOT EXISTS calls (
    call_id              INTEGER PRIMARY KEY,   -- earningscalls.dev earnings call id
    call_key             TEXT,                  -- e.g. NVDA_2025Q3 (SDS-style id)
    ticker               TEXT NOT NULL,
    company_name         TEXT,
    transcript_title     TEXT,
    event_type           TEXT,
    fiscal_year          INTEGER,
    fiscal_quarter       INTEGER,
    event_datetime_raw   TEXT,
    event_datetime_utc   TEXT,
    event_datetime_et    TEXT,
    call_date_et         TEXT,
    call_session         TEXT,                  -- pre_market / intraday / after_close / unknown
    mentions_ai          INTEGER,               -- matched the API full-text AI query
    ai_query             TEXT,
    text_source          TEXT,                  -- speakers / transcript / none
    has_speaker_labels   INTEGER,
    n_segments           INTEGER,
    n_sentences          INTEGER,
    n_words              INTEGER,
    n_words_ex_operator  INTEGER,
    n_ai_sentences       INTEGER,
    status               TEXT,                  -- ok / no_text / error / pending
    status_detail        TEXT,
    pulled_at            TEXT,
    FOREIGN KEY (ticker) REFERENCES companies(ticker)
);
CREATE INDEX IF NOT EXISTS ix_calls_ticker ON calls(ticker, call_date_et);

CREATE TABLE IF NOT EXISTS segments (
    segment_id       TEXT PRIMARY KEY,          -- <call_id>-<order>
    call_id          INTEGER NOT NULL,
    component_order  INTEGER,
    speaker_name     TEXT,
    speaker_type     TEXT,                      -- executive / analyst / operator / ...
    speaker_title    TEXT,
    section          TEXT,                      -- prepared_remarks / qa
    text_raw         TEXT,
    text_clean       TEXT,
    n_words          INTEGER,
    is_operator      INTEGER,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);
CREATE INDEX IF NOT EXISTS ix_segments_call ON segments(call_id);

CREATE TABLE IF NOT EXISTS sentences (
    sentence_id   TEXT PRIMARY KEY,             -- <call_id>-<order>-<idx>, stable across runs
    call_id       INTEGER NOT NULL,
    segment_id    TEXT NOT NULL,
    sent_idx      INTEGER,
    section       TEXT,
    speaker_type  TEXT,
    text          TEXT,
    n_words       INTEGER,
    is_boilerplate INTEGER,                     -- safe-harbor / forward-looking disclaimer
    has_ai_term   INTEGER,
    ai_terms      TEXT,                         -- pipe-separated matched terms
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
);
CREATE INDEX IF NOT EXISTS ix_sentences_call ON sentences(call_id);

CREATE TABLE IF NOT EXISTS market_data (
    ticker     TEXT NOT NULL,                   -- transcript ticker (yf symbol in companies)
    date       TEXT NOT NULL,
    open       REAL, high REAL, low REAL, close REAL, adj_close REAL,
    volume     REAL,
    dividends  REAL,
    splits     REAL,
    ret        REAL,                            -- simple return on adj_close
    log_ret    REAL,
    flag       TEXT,                            -- data-quality flags, pipe separated
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS yf_earnings_dates (
    ticker             TEXT NOT NULL,
    earnings_datetime  TEXT NOT NULL,
    earnings_date_et   TEXT,
    eps_estimate       REAL,
    eps_reported       REAL,
    surprise_pct       REAL,
    PRIMARY KEY (ticker, earnings_datetime)
);

-- event_returns is derived data: rebuilt with pandas.to_sql by events.py on every
-- run, with one column per configured window (car_mkt_0_1, car_mm_m1_1, ...).

CREATE TABLE IF NOT EXISTS run_log (
    run_id        TEXT,
    stage         TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    api_requests  INTEGER,
    cache_hits    INTEGER,
    status        TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS annotation_samples (
    batch_id       TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    requested_size INTEGER NOT NULL,
    actual_size    INTEGER NOT NULL,
    double_fraction REAL NOT NULL,
    seed           INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS annotation_sample_items (
    batch_id       TEXT NOT NULL,
    sentence_id    TEXT NOT NULL,
    sample_index   INTEGER NOT NULL,
    double_label   INTEGER NOT NULL,
    context_text   TEXT NOT NULL,
    PRIMARY KEY (batch_id, sentence_id),
    UNIQUE (batch_id, sample_index),
    FOREIGN KEY (batch_id) REFERENCES annotation_samples(batch_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id)
);

CREATE TABLE IF NOT EXISTS annotations (
    batch_id       TEXT NOT NULL,
    sentence_id    TEXT NOT NULL,
    annotator      TEXT NOT NULL,
    label          INTEGER NOT NULL CHECK (label BETWEEN 0 AND 3),
    notes          TEXT,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (batch_id, sentence_id, annotator),
    FOREIGN KEY (batch_id, sentence_id)
        REFERENCES annotation_sample_items(batch_id, sentence_id)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = OFF")   # stages can run independently
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, table: str, rows: Sequence[dict], conflict_cols: Iterable[str] | None = None) -> int:
    """INSERT OR REPLACE a list of dicts (columns taken from the table)."""
    if not rows:
        return 0
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    use = [c for c in cols if any(c in r for r in rows)]
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(use)}) VALUES ({','.join('?' * len(use))})"
    con.executemany(sql, [tuple(_py(r.get(c)) for c in use) for r in rows])
    return len(rows)


def update_partial(con: sqlite3.Connection, table: str, key: str, rows: Sequence[dict]) -> None:
    """Update only the given columns of existing rows (insert when missing)."""
    for r in rows:
        cols = [c for c in r if c != key]
        exists = con.execute(f"SELECT 1 FROM {table} WHERE {key}=?", (r[key],)).fetchone()
        if exists:
            if cols:
                con.execute(
                    f"UPDATE {table} SET {', '.join(f'{c}=?' for c in cols)} WHERE {key}=?",
                    [_py(r[c]) for c in cols] + [r[key]],
                )
        else:
            upsert(con, table, [r])


def replace_where(con: sqlite3.Connection, table: str, where: str, params: tuple, rows: Sequence[dict]) -> None:
    con.execute(f"DELETE FROM {table} WHERE {where}", params)
    upsert(con, table, rows)


def read_df(con: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def _py(v):
    """Convert numpy/pandas scalars and NaN to plain Python for sqlite3."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v
