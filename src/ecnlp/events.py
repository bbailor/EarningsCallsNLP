"""Align each call to the trading calendar and measure the market reaction.

This is the bridge that lets NLP scores be cross-referenced with Yahoo data
later: one row per call with day 0, market-adjusted and market-model
cumulative abnormal returns over the configured windows, abnormal volume,
and Yahoo's EPS surprise as a control.

Day 0 is the first session in which the market could trade on the call:
* after_close (>= 16:00 ET)  -> next trading day
* pre_market / intraday      -> same day (or next trading day if a holiday)
* unknown time               -> same calendar date rule, flagged; prefer the
                                [-1, +1] window for these calls
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)


def _wname(a: int, b: int) -> str:
    f = lambda x: f"m{abs(x)}" if x < 0 else str(x)
    return f"{f(a)}_{f(b)}"


def find_day0(calendar: pd.DatetimeIndex, call_date: str, session: str) -> int | None:
    d = pd.Timestamp(call_date)
    side = "right" if session == "after_close" else "left"
    pos = calendar.searchsorted(d, side=side)
    return int(pos) if pos < len(calendar) else None


def compute_event_returns(cfg: Config, con: sqlite3.Connection) -> pd.DataFrame:
    ev = cfg.events
    bench = str((cfg.market.get("benchmarks") or ["SPY"])[0]).upper()
    windows = [tuple(map(int, w)) for w in ev.get("windows", [[0, 1], [-1, 1], [0, 3]])]
    est_a, est_b = map(int, ev.get("estimation_window", [-250, -11]))
    min_obs = int(ev.get("min_estimation_obs", 100))
    vol_a, vol_b = map(int, ev.get("volume_baseline_window", [-60, -11]))
    match_days = int(ev.get("earnings_date_match_days", 3))

    mkt = db.read_df(con, "SELECT date, ret FROM market_data WHERE ticker=? ORDER BY date", (bench,))
    if mkt.empty:
        log.error("No benchmark (%s) prices; run the market stage first", bench)
        return pd.DataFrame()
    mkt["date"] = pd.to_datetime(mkt["date"])
    cal = pd.DatetimeIndex(mkt["date"])
    rm = mkt.set_index("date")["ret"]

    calls = db.read_df(con, "SELECT call_id, ticker, call_date_et, call_session FROM calls "
                            "WHERE call_date_et IS NOT NULL ORDER BY ticker, call_date_et")
    eds = db.read_df(con, "SELECT ticker, earnings_date_et, eps_estimate, eps_reported, surprise_pct FROM yf_earnings_dates")
    if not eds.empty:
        eds["earnings_date_et"] = pd.to_datetime(eds["earnings_date_et"])

    out: list[dict[str, Any]] = []
    price_cache: dict[str, pd.DataFrame] = {}
    for c in calls.itertuples(index=False):
        flags: list[str] = []
        row: dict[str, Any] = {"call_id": c.call_id, "ticker": c.ticker, "benchmark": bench,
                               "call_date_et": c.call_date_et, "call_session": c.call_session}
        if c.call_session == "unknown":
            flags.append("session_unknown")

        if c.ticker not in price_cache:
            p = db.read_df(con, "SELECT date, close, ret, volume FROM market_data WHERE ticker=? ORDER BY date", (c.ticker,))
            p["date"] = pd.to_datetime(p["date"])
            price_cache[c.ticker] = p.set_index("date").reindex(cal)  # align to market calendar
        px = price_cache[c.ticker]
        if px["close"].notna().sum() == 0:
            row["quality_flags"] = "no_price_data"
            out.append(row)
            continue

        d0 = find_day0(cal, c.call_date_et, c.call_session)
        if d0 is None:
            row["quality_flags"] = "|".join(flags + ["call_after_last_price_date"])
            out.append(row)
            continue
        row["day0"] = cal[d0].strftime("%Y-%m-%d")
        ri = px["ret"].to_numpy()
        rmi = rm.to_numpy()

        def sl(a: int, b: int) -> slice:
            return slice(d0 + a, d0 + b + 1)

        # market model on the estimation window
        lo, hi = d0 + est_a, d0 + est_b
        alpha = beta = None
        if lo >= 0:
            y, x = ri[lo:hi + 1], rmi[lo:hi + 1]
            ok = ~(np.isnan(y) | np.isnan(x))
            row["n_est_obs"] = int(ok.sum())
            if ok.sum() >= min_obs:
                beta = float(np.cov(y[ok], x[ok], ddof=1)[0, 1] / np.var(x[ok], ddof=1))
                alpha = float(y[ok].mean() - beta * x[ok].mean())
            else:
                flags.append("short_estimation_window")
        else:
            row["n_est_obs"] = 0
            flags.append("short_estimation_window")
        row["alpha"], row["beta"] = alpha, beta

        win_json = {}
        for a, b in windows:
            name = _wname(a, b)
            s = sl(a, b)
            if s.start < 0 or s.stop > len(cal):
                flags.append(f"window_{name}_out_of_range")
                continue
            r, m = ri[s], rmi[s]
            if np.isnan(r).any() or np.isnan(m).any():
                flags.append(f"window_{name}_missing_days")
                continue
            row[f"bhr_{name}"] = float(np.prod(1 + r) - 1)             # raw buy-and-hold return
            row[f"car_mkt_{name}"] = float(np.sum(r - m))              # market-adjusted CAR
            if beta is not None:
                row[f"car_mm_{name}"] = float(np.sum(r - (alpha + beta * m)))  # market-model CAR
            win_json[name] = row[f"car_mkt_{name}"]
        row["window_cols"] = json.dumps(win_json)

        vol = px["volume"].to_numpy()
        if d0 + vol_a >= 0 and d0 + 2 <= len(cal):
            base = np.nanmean(vol[d0 + vol_a:d0 + vol_b + 1])
            evv = np.nanmean(vol[d0:d0 + 2])
            row["abnormal_volume_0_1"] = float(evv / base - 1) if base and base > 0 else None
        if d0 >= 1:
            row["pre_event_close"] = float(px["close"].iloc[d0 - 1]) if pd.notna(px["close"].iloc[d0 - 1]) else None

        if not eds.empty:
            e = eds[eds["ticker"] == c.ticker]
            if len(e):
                diff = (e["earnings_date_et"] - pd.Timestamp(c.call_date_et)).dt.days
                j = diff.abs().idxmin()
                if abs(diff[j]) <= match_days:
                    row.update({"eps_estimate": e.at[j, "eps_estimate"], "eps_reported": e.at[j, "eps_reported"],
                                "surprise_pct": e.at[j, "surprise_pct"], "yf_date_match_days": int(diff[j])})
                else:
                    flags.append("no_yahoo_earnings_date_match")
        row["quality_flags"] = "|".join(flags) or None
        out.append(row)

    df = pd.DataFrame(out)
    df.to_sql("event_returns", con, if_exists="replace", index=False)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_event_returns_call ON event_returns(call_id)")
    con.commit()
    cars = df.filter(like="car_mkt_")
    complete = int(cars.notna().all(axis=1).sum()) if len(df) and cars.shape[1] else 0
    log.info("Event returns: %d calls, %d with every window complete", len(df), complete)
    return df
