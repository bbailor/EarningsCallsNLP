"""Call timestamp handling.

Getting the call time right matters for the event study: a call at 5 pm ET
can only move the stock on the *next* trading day.
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd

ET = "America/New_York"


def parse_event_time(raw: Any, naive_tz: str = "UTC") -> tuple[pd.Timestamp | None, bool]:
    """Return (UTC timestamp, time_known).

    `time_known` is False when the value is date-only, or exactly midnight UTC
    (a common placeholder). Converting such a value to ET would wrongly push
    the call to the previous evening, so callers should use the raw date.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
        return None, False
    s = str(raw).strip()
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError):
        return None, False
    date_only = len(s) <= 10
    if ts.tzinfo is None:
        ts = ts.tz_localize(naive_tz)
    ts = ts.tz_convert("UTC")
    placeholder = ts.hour == 0 and ts.minute == 0 and ts.second == 0
    return ts, not (date_only or placeholder)


def classify_session(ts_utc: pd.Timestamp | None, time_known: bool, raw: Any,
                     open_hm: str = "09:30", close_hm: str = "16:00") -> dict[str, Any]:
    """Return ET datetime, ET calendar date and the trading session of the call."""
    if ts_utc is None:
        return {"event_datetime_utc": None, "event_datetime_et": None, "call_date_et": None, "call_session": "unknown"}
    if not time_known:
        d = str(raw)[:10]
        return {
            "event_datetime_utc": ts_utc.isoformat(),
            "event_datetime_et": None,
            "call_date_et": d,
            "call_session": "unknown",
        }
    et = ts_utc.tz_convert(ET)
    o = time.fromisoformat(open_hm)
    c = time.fromisoformat(close_hm)
    t = et.time()
    session = "pre_market" if t < o else ("intraday" if t < c else "after_close")
    return {
        "event_datetime_utc": ts_utc.isoformat(),
        "event_datetime_et": et.isoformat(),
        "call_date_et": et.date().isoformat(),
        "call_session": session,
    }
