"""Shared fixtures: a fake earningscalls.dev API and fake yfinance functions.

Payload shapes follow the official OpenAPI spec and doc examples
(earnings list fields, by_ticker `results[].earnings_ids`, speaker segments
with text_content / component_order / speaker_name / speaker_type).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
import responses
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE = "https://earningscalls.dev/api/v1"

# ticker -> list of (id, event_date_time, FY, Q, mentions_ai)
CALLS = {
    "NVDA": [(1001, "2025-05-28T21:00:00.000Z", 2026, 1, True),
             (1002, "2025-08-27T21:00:00.000Z", 2026, 2, True)],
    "CAT":  [(2001, "2025-04-30T12:30:00Z", 2025, 1, False),
             (2002, "2025-08-05T12:30:00Z", 2025, 2, True)],
    "BRK.B": [(3001, "2025-05-03", None, None, True)],   # date-only, title-parsed fiscal period
}
TITLES = {3001: "Berkshire Hathaway Q1 2025 Earnings Call"}


def segments_for(cid: int) -> list[dict]:
    segs = [
        {"component_order": 0, "speaker_name": "Operator", "speaker_type": "operator",
         "text_content": "Good afternoon. [Operator Instructions] This call is being recorded."},
        {"component_order": 1, "speaker_name": "Jane CFO", "speaker_type": "executive",
         "text_content": "Today's remarks include forward-looking statements. Actual results may differ materially. "
                         "Revenue grew 12% to $4.5 billion. We deployed an AI-based forecasting system across our "
                         "warehouses and reduced inventory by 8%."},
        {"component_order": 2, "speaker_name": "Operator", "speaker_type": "operator",
         "text_content": "Our first question comes from Bob."},
        {"component_order": 3, "speaker_name": "Bob Analyst", "speaker_type": "analyst",
         "text_content": "Thanks. Can you talk about generative AI demand in the U.S. market?"},
        {"component_order": 4, "speaker_name": "John CEO", "speaker_type": "executive",
         "text_content": "Sure. Mr. Smith's team at Acme Inc. is piloting copilots. Machine learning now "
                         "drives 30% of pricing decisions."},
    ]
    if cid == 3001:   # an old "standard" transcript without speaker identification
        return []
    return segs


def _qs(request) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}


def install_fake_api(rsps: responses.RequestsMock, counter: dict) -> None:
    def count(fn):
        def wrapped(request):
            counter["n"] = counter.get("n", 0) + 1
            assert request.headers.get("X-API-Key") == "test-key"
            return fn(request)
        return wrapped

    def me(request):
        return 200, {}, '{"data": {"tier": "ultra", "requests_used": 120, "requests_limit": 20000}}'

    def by_ticker(request):
        q = _qs(request)
        assert q["q"] and q["tickers"]
        res = []
        for t in q["tickers"].split(","):
            ids = [str(c[0]) for c in CALLS.get(t, []) if c[4]]
            if ids:
                res.append({"ticker": f"{t}:US" if t == "NVDA" else t, "company_name": t,
                            "calls_matched": len(ids), "earnings_ids": ids})
        import json
        return 200, {}, json.dumps({"total_companies": len(res), "total_calls": sum(len(r["earnings_ids"]) for r in res),
                                    "results": res})

    def earnings(request):
        import json
        q = _qs(request)
        t = q["ticker"]
        rows = []
        for cid, dt, fy, fq, _ in CALLS.get(t, []):
            rows.append({"id": cid, "company_name": f"{t} Corp", "transcript_title": TITLES.get(cid, f"{t} Q{fq} {fy} Earnings Call"),
                         "event_type": "earnings", "event_date_time": dt, "sector": "Technology", "industry": "Semis",
                         "stock_symbol": t, "company_ticker": t, "exchange": "NASDAQ", "mic": "XNAS", "country": "US",
                         "currency": "USD", "fiscal_year": fy, "fiscal_quarter": fq, "issuer_id": "a" * 32,
                         "isin": None, "timezone": "America/New_York", "matched_alias": t})
        # A foreign non-earnings event the AI search might also return is simply absent here.
        return 200, {}, json.dumps({"data": rows, "pagination": {"page": 1, "limit": 100, "total": len(rows), "total_pages": 1},
                                    "ambiguous": False, "issuer_count": 1})

    def speakers(request):
        import json
        cid = int(re.search(r"/speakers/(\d+)", request.url).group(1))
        segs = segments_for(cid)
        return 200, {}, json.dumps({"data": segs, "pagination": {"page": 1, "limit": 200, "total": len(segs), "total_pages": 1}})

    def transcript(request):
        import json
        return 200, {}, json.dumps({"data": {"id": 3001, "transcript_text":
                                             "Welcome to the meeting. Our insurance float grew. We do not use AI much.",
                                             "next_offset": None}})

    rsps.add_callback(responses.GET, f"{BASE}/me", callback=count(me))
    rsps.add_callback(responses.GET, re.compile(rf"{BASE}/search/by_ticker.*"), callback=count(by_ticker))
    rsps.add_callback(responses.GET, re.compile(rf"{BASE}/earnings/\?.*"), callback=count(earnings))
    rsps.add_callback(responses.GET, re.compile(rf"{BASE}/speakers/\d+.*"), callback=count(speakers))
    rsps.add_callback(responses.GET, re.compile(rf"{BASE}/transcripts/\d+.*"), callback=count(transcript))


def fake_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    if symbol == "DELISTED":
        raise RuntimeError("YFPricesMissingError: possibly delisted; no price data found")
    idx = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), tz="America/New_York")
    rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
    r = rng.normal(0.0004, 0.015, len(idx))
    px = 100 * np.cumprod(1 + r)
    return pd.DataFrame({"Open": px, "High": px * 1.01, "Low": px * 0.99, "Close": px, "Adj Close": px * 0.98,
                         "Volume": rng.integers(1e6, 5e6, len(idx)).astype(float),
                         "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)


def fake_info(symbol: str) -> dict:
    return {"longName": f"{symbol} Inc", "sector": "Technology", "industry": "Semiconductors",
            "marketCap": 1e12, "sharesOutstanding": 2.4e10, "quoteType": "EQUITY"}


def fake_earnings_dates(symbol: str, limit: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2025-05-28 16:00", "2025-08-27 16:00", "2025-08-05 07:00"], tz="America/New_York",
                           name="Earnings Date")
    return pd.DataFrame({"EPS Estimate": [0.73, 1.01, 4.9], "Reported EPS": [0.81, 1.05, 4.72],
                         "Surprise(%)": [10.96, 3.96, -3.67]}, index=idx)


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    base = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    base.update({"data_dir": str(tmp_path / "data"), "tickers": ["NVDA", "CAT", "BRK.B"],
                 "date_from": "2025-01-01", "date_to": "2025-09-30"})
    base["market"]["pause_between_tickers_sec"] = 0
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base))
    monkeypatch.setenv("EARNINGSCALLS_API_KEY", "test-key")
    monkeypatch.delenv("ECNLP_DATA_DIR", raising=False)
    return p
