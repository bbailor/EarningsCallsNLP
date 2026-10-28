from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
import responses

from conftest import BASE, fake_earnings_dates, fake_history, fake_info, install_fake_api
from ecnlp import db
from ecnlp.cleaning import (assign_sections, build_ai_matcher, is_boilerplate, normalize_text,
                            parse_fiscal_period, split_sentences)
from ecnlp.config import load_config
from ecnlp.earningscalls_client import AuthError, EarningsCallsClient, QuotaExceeded, parse_usage
from ecnlp.events import compute_event_returns, find_day0
from ecnlp.exports import export_all
from ecnlp.market_data import MarketDataPuller, clean_prices, pull_market_data, yf_symbol
from ecnlp.prepare import prepare_text
from ecnlp.timeutils import classify_session, parse_event_time
from ecnlp.transcripts import pull_transcripts


# ------------------------------------------------------------------ cleaning
def test_sentence_splitter_handles_abbreviations_and_decimals():
    text = "Revenue was $4.5 billion, up 12%. Mr. Smith of Acme Inc. joined the U.S. team. Next question."
    assert split_sentences(text) == [
        "Revenue was $4.5 billion, up 12%.",
        "Mr. Smith of Acme Inc. joined the U.S. team.",
        "Next question.",
    ]


def test_ai_matcher_word_boundaries_and_case():
    m = build_ai_matcher(["AI", r"A\.I\.", "ML", "LLMs?"], ["machine learning", "generative ai", "copilots?"])
    assert m("Our AI-powered tools and A.I. roadmap") == ["AI", "A.I."]
    assert m("We said ai and Thailand and SAID") == []          # case-sensitive AI, no substrings
    assert m("Generative AI and Machine Learning") == ["AI", "generative ai", "machine learning"]
    assert m("200 ml of solution") == []
    assert m("New Copilots for LLMs") == ["LLMs", "copilots"]


def test_boilerplate_and_normalize():
    assert is_boilerplate("This call contains forward-looking statements.")
    assert not is_boilerplate("Revenue grew 10%.")
    assert normalize_text("We’re  growing — fast [Operator Instructions] ok") == "We're growing - fast ok"


def test_sections_start_at_first_analyst():
    assert assign_sections(["operator", "executive", "operator", "analyst", "executive"]) == \
        ["prepared_remarks", "prepared_remarks", "qa", "qa", "qa"]
    assert assign_sections(["executive", "executive"]) == ["prepared_remarks"] * 2


def test_fiscal_period_from_title():
    assert parse_fiscal_period("NVIDIA Q3 FY2026 Earnings Call") == (2026, 3)
    assert parse_fiscal_period("Acme 2025 Q2 results") == (2025, 2)
    assert parse_fiscal_period("Third Quarter Fiscal 2024 Call") == (2024, 3)
    assert parse_fiscal_period("Investor Day") == (None, None)


# ----------------------------------------------------------------- timing
def test_session_classification():
    ts, known = parse_event_time("2025-05-28T21:00:00Z")
    assert known and classify_session(ts, known, None)["call_session"] == "after_close"
    ts, known = parse_event_time("2025-04-30T12:30:00Z")
    r = classify_session(ts, known, None)
    assert r["call_session"] == "pre_market" and r["call_date_et"] == "2025-04-30"
    # date-only must NOT be shifted to the previous evening in ET
    ts, known = parse_event_time("2025-05-03")
    r = classify_session(ts, known, "2025-05-03")
    assert not known and r["call_date_et"] == "2025-05-03" and r["call_session"] == "unknown"


def test_day0_rules():
    cal = pd.DatetimeIndex(pd.to_datetime(["2025-05-27", "2025-05-28", "2025-05-29"]))
    assert find_day0(cal, "2025-05-28", "after_close") == 2
    assert find_day0(cal, "2025-05-28", "pre_market") == 1
    assert find_day0(cal, "2025-05-25", "intraday") == 0      # weekend -> next session


# ------------------------------------------------------------------ client
@responses.activate
def test_client_404_counts_and_returns_none(tmp_path):
    responses.add(responses.GET, f"{BASE}/earnings/99", status=404, json={"error": "Not Found", "message": "no"})
    c = EarningsCallsClient("k", cache_dir=tmp_path)
    assert c.get("earnings/99") is None
    assert c.requests_made == 1          # 404s count toward quota per the docs


@responses.activate
def test_client_monthly_quota_raises(tmp_path):
    responses.add(responses.GET, f"{BASE}/me", status=429,
                  json={"error": "Too Many Requests", "message": "Monthly quota exceeded. Resets 2026-11-01"})
    c = EarningsCallsClient("k", cache_dir=tmp_path)
    with pytest.raises(QuotaExceeded):
        c.me()
    assert c.requests_made == 0          # 429 does not count


@responses.activate
def test_client_retries_5xx_then_caches(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    responses.add(responses.GET, f"{BASE}/speakers/5", status=502)
    responses.add(responses.GET, f"{BASE}/speakers/5", json={"data": [{"text_content": "hi"}]})
    c = EarningsCallsClient("k", cache_dir=tmp_path)
    assert c.speakers(5)["data"][0]["text_content"] == "hi"
    assert c.speakers(5)["data"][0]["text_content"] == "hi"   # second call from cache
    assert len(responses.calls) == 2 and c.cache_hits == 1


def test_client_requires_key():
    with pytest.raises(AuthError):
        EarningsCallsClient("")


def test_parse_usage_variants():
    assert parse_usage({"data": {"requests_used": 10, "requests_limit": 100}})["remaining"] == 90
    assert parse_usage({"usage": {"remaining": 5}})["remaining"] == 5


# ----------------------------------------------------------- market cleaning
def test_clean_prices_and_symbols():
    raw = fake_history("NVDA", "2025-01-01", "2025-02-01")
    df = clean_prices(raw)
    assert list(df.columns)[:3] == ["date", "open", "high"]
    assert df["date"].is_monotonic_increasing and df["ret"].iloc[1:].notna().all()
    assert yf_symbol("BRK.B") == "BRK-B" and yf_symbol("AAPL:US") == "AAPL"


# --------------------------------------------------------------- end to end
def test_full_pipeline_end_to_end(cfg_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = load_config(cfg_path)
    cfg.paths.ensure()
    con = db.connect(cfg.paths.db)
    counter: dict = {}

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        install_fake_api(rsps, counter)
        client = EarningsCallsClient("test-key", cache_dir=cfg.paths.raw_ec)
        stats = pull_transcripts(cfg, client, con, remaining_quota=10000)
        first_run_requests = counter["n"]

        # second run: everything cached -> zero API requests
        client2 = EarningsCallsClient("test-key", cache_dir=cfg.paths.raw_ec)
        pull_transcripts(cfg, client2, con, remaining_quota=10000)
        assert counter["n"] == first_run_requests
        assert client2.requests_made == 0

    # only AI-matching calls kept: CAT 2001 has no AI mention
    calls = db.read_df(con, "SELECT * FROM calls ORDER BY call_id")
    assert set(calls["call_id"]) == {1001, 1002, 2002, 3001}
    assert stats["ok"] == 4
    nv = calls.set_index("call_id").loc[1001]
    assert nv["call_session"] == "after_close" and nv["call_key"] == "NVDA_2026Q1"
    brk = calls.set_index("call_id").loc[3001]
    assert brk["text_source"] == "transcript" and brk["call_key"] == "BRK.B_2025Q1" and brk["call_session"] == "unknown"
    assert calls.set_index("call_id").loc[1001, "has_speaker_labels"] == 1

    prepare_text(cfg, con)
    sents = db.read_df(con, "SELECT * FROM sentences WHERE call_id=1001 ORDER BY segment_id, sent_idx")
    assert not sents["speaker_type"].eq("operator").any()                 # operator turns dropped
    assert sents["is_boilerplate"].sum() >= 1                              # safe-harbor sentence flagged
    assert sents.loc[sents["section"] == "qa", "text"].str.contains("generative AI").any()
    assert sents["sentence_id"].is_unique

    puller = MarketDataPuller(cfg, fetch_history=fake_history, fetch_info=fake_info,
                              fetch_earnings_dates=fake_earnings_dates)
    mstats = pull_market_data(cfg, con, puller)
    assert mstats["failed"] == []
    calls_before = puller.network_calls
    pull_market_data(cfg, con, puller)            # second run uses the cache
    assert puller.network_calls == calls_before
    assert con.execute("SELECT yf_symbol FROM companies WHERE ticker='BRK.B'").fetchone()[0] == "BRK-B"

    er = compute_event_returns(cfg, con).set_index("call_id")
    assert er.loc[1001, "day0"] == "2025-05-29"     # after-close call reacts next day
    assert er.loc[2002, "day0"] == "2025-08-05"     # pre-market call reacts same day
    assert pd.notna(er.loc[1001, "car_mkt_0_1"]) and pd.notna(er.loc[1001, "car_mm_m1_1"])
    assert er.loc[1001, "surprise_pct"] == pytest.approx(10.96)
    assert "session_unknown" in er.loc[3001, "quality_flags"]

    files = export_all(cfg, con)
    master = pd.read_csv(files["calls_master"])
    assert {"car_mkt_0_1", "ai_sentences_per_1000_words", "yf_market_cap"} <= set(master.columns)
    ctx = pd.read_parquet(files["ai_contexts"])
    assert len(ctx) > 0 and ctx["context_text"].str.len().gt(ctx["sentence"].str.len()).any()
    cov = pd.read_csv(files["coverage"]).set_index("ticker")
    assert cov.loc["CAT", "calls_in_api"] == 2 and cov.loc["CAT", "calls_selected"] == 1
    con.close()


def test_market_failure_is_recorded(cfg_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = load_config(cfg_path, {"tickers": ["DELISTED"]})
    cfg.paths.ensure()
    con = db.connect(cfg.paths.db)
    puller = MarketDataPuller(cfg, fetch_history=fake_history, fetch_info=fake_info,
                              fetch_earnings_dates=fake_earnings_dates)
    stats = pull_market_data(cfg, con, puller)
    assert stats["failed"] == ["DELISTED"]
    assert con.execute("SELECT yf_status FROM companies WHERE ticker='DELISTED'").fetchone()[0].startswith("failed")


def test_cli_all_stages(cfg_path, monkeypatch):
    """Runs the real CLI entry point (all stages) against the fake API and fake yfinance."""
    import ecnlp.market_data as md
    from ecnlp.pipeline import main

    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(md.MarketDataPuller.__init__, "__defaults__",
                        (fake_history, fake_info, fake_earnings_dates))
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        install_fake_api(rsps, {})
        assert main(["all", "--config", str(cfg_path)]) == 0
    cfg = load_config(cfg_path)
    con = sqlite3.connect(cfg.paths.db)
    log = pd.read_sql_query("SELECT stage, status FROM run_log", con)
    assert list(log["stage"]) == ["transcripts", "market", "prepare", "events", "export"]
    assert (log["status"] == "ok").all()
    assert (cfg.paths.processed / "nlp_sentences.parquet").exists()
    assert any(cfg.paths.logs.glob("all_*.log"))
