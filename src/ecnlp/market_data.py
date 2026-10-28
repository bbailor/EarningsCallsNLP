"""NLP-02: pull market data for the same tickers from Yahoo Finance (yfinance).

Per ticker:
* daily OHLCV with both raw Close and Adj Close (auto_adjust=False; the
  library default changed to True in 0.2.54, which silently drops Adj Close)
* dividends / splits
* a company-info snapshot (sector, industry, market cap, shares outstanding)
* Yahoo's earnings-date table (EPS estimate, reported EPS, surprise %), a
  useful control for the regressions because EPS surprise drives returns

Benchmarks (SPY by default) are pulled the same way for abnormal returns.

Caching: prices are stored as CSV in raw/yfinance. A re-run within
`price_cache_days` skips the ticker; after that only the missing tail is
fetched (incremental update), so a second run costs almost nothing.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)

PRICE_COLS = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
              "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividends",
              "Stock Splits": "splits"}


def yf_symbol(ticker: str, overrides: dict | None = None) -> str:
    overrides = {str(k).upper(): v for k, v in (overrides or {}).items()}
    t = ticker.split(":")[0].upper()
    if t in overrides:
        return overrides[t]
    return t.replace(".", "-").replace("/", "-")


# ------------------------------------------------------------ yfinance calls
def _default_fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=False,
                                     actions=True, raise_errors=True, timeout=30)


def _default_fetch_info(symbol: str) -> dict:
    import yfinance as yf

    return yf.Ticker(symbol).info or {}


def _default_fetch_earnings_dates(symbol: str, limit: int) -> pd.DataFrame | None:
    import yfinance as yf

    return yf.Ticker(symbol).get_earnings_dates(limit=min(int(limit), 100))


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__
    return name == "YFRateLimitError" or "Too Many Requests" in str(exc) or "429" in str(exc)


def _is_no_data(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in ("YFPricesMissingError", "YFTzMissingError", "YFTickerMissingError") or \
        "delisted" in str(exc).lower() or "no data found" in str(exc).lower()


def _with_retries(fn: Callable, *args, retries: int = 4, what: str = "") -> Any:
    for attempt in range(retries + 1):
        try:
            return fn(*args)
        except Exception as exc:  # yfinance raises many types
            if _is_no_data(exc):
                raise
            if attempt == retries:
                raise
            wait = (30 if _is_rate_limit(exc) else 3) * (2 ** attempt)
            log.warning("%s failed (%s: %s); retry %d in %ss", what, type(exc).__name__, exc, attempt + 1, wait)
            time.sleep(wait)


# ---------------------------------------------------------------- cleaning
def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize yfinance output into one row per trading date with QA flags."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", *PRICE_COLS.values(), "ret", "log_ret", "flag"])
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # yf.download style; flatten
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=PRICE_COLS)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)  # exchange-local dates; drop tz before normalizing
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in PRICE_COLS.values():
        if c not in df.columns:
            df[c] = 0.0 if c in ("dividends", "splits") else np.nan
    if df["adj_close"].isna().all():
        df["adj_close"] = df["close"]
    df = df[list(PRICE_COLS.values())].astype(float)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    df["ret"] = df["adj_close"].pct_change()
    df["log_ret"] = np.log(df["adj_close"]).diff()
    flags = pd.Series("", index=df.index)
    flags[df["volume"].fillna(0) <= 0] += "zero_volume|"
    flags[df["ret"].abs() > 0.5] += "extreme_return|"
    flags[df["close"] < 1] += "penny_price|"
    flags[(df["high"] < df["low"])] += "high_lt_low|"
    df["flag"] = flags.str.rstrip("|").replace("", None)
    df.index.name = "date"
    return df.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))


def clean_earnings_dates(raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("America/New_York")
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if "estimate" in cl:
            colmap[c] = "eps_estimate"
        elif "reported" in cl:
            colmap[c] = "eps_reported"
        elif "surprise" in cl:
            colmap[c] = "surprise_pct"
    df = df.rename(columns=colmap)
    for c in ("eps_estimate", "eps_reported", "surprise_pct"):
        if c not in df.columns:
            df[c] = np.nan
    out = pd.DataFrame({
        "earnings_datetime": df.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "earnings_date_et": df.index.tz_convert("America/New_York").strftime("%Y-%m-%d"),
        "eps_estimate": pd.to_numeric(df["eps_estimate"], errors="coerce").values,
        "eps_reported": pd.to_numeric(df["eps_reported"], errors="coerce").values,
        "surprise_pct": pd.to_numeric(df["surprise_pct"], errors="coerce").values,
    })
    return out.drop_duplicates("earnings_datetime")


# ------------------------------------------------------------------ caching
def _meta_path(cache: Path, symbol: str) -> Path:
    return cache / f"{symbol}_meta.json"


def _prices_path(cache: Path, symbol: str) -> Path:
    return cache / f"{symbol}_prices.csv"


def _load_cached(cache: Path, symbol: str) -> tuple[pd.DataFrame | None, dict]:
    p, m = _prices_path(cache, symbol), _meta_path(cache, symbol)
    if not p.exists() or not m.exists():
        return None, {}
    try:
        return pd.read_csv(p), json.loads(m.read_text())
    except (ValueError, json.JSONDecodeError):
        log.warning("Corrupt price cache for %s; refetching", symbol)
        return None, {}


def _age_days(iso: str | None) -> float:
    if not iso:
        return 1e9
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 86400


class MarketDataPuller:
    """Wraps the yfinance calls so tests can inject fakes."""

    def __init__(self, cfg: Config, fetch_history: Callable = _default_fetch_history,
                 fetch_info: Callable = _default_fetch_info,
                 fetch_earnings_dates: Callable = _default_fetch_earnings_dates) -> None:
        self.cfg = cfg
        self.m = cfg.market
        self.cache = cfg.paths.raw_yf
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fetch_history = fetch_history
        self.fetch_info = fetch_info
        self.fetch_earnings_dates = fetch_earnings_dates
        self.retries = int(self.m.get("max_retries", 4))
        self.network_calls = 0

    def price_range(self) -> tuple[str, str]:
        start = (date.fromisoformat(self.cfg.date_from) - timedelta(days=int(self.m.get("history_buffer_days", 400))))
        # +45 days after date_to so post-event windows (up to +30 trading days) are covered
        end = min(date.fromisoformat(self.cfg.date_to) + timedelta(days=45), date.today() + timedelta(days=1))
        return start.isoformat(), end.isoformat()

    def prices(self, symbol: str) -> tuple[pd.DataFrame, str]:
        """Return (clean prices, status) using the cache when it is fresh and covers the range."""
        start, end = self.price_range()
        cached, meta = _load_cached(self.cache, symbol)
        usable = cached is not None and len(cached) > 0 and meta.get("start", "9999") <= start
        if usable:
            fresh = _age_days(meta.get("pulled_at")) < float(self.m.get("price_cache_days", 1))
            # A fully historical range (ended well before the pull) never changes.
            pulled_day = (meta.get("pulled_at") or "")[:10]
            historical = meta.get("end", "") >= end and end < pulled_day
            if fresh or historical:
                return cached, "cached"
        if usable:
            # incremental: refetch from a few days before the last cached date
            last = cached["date"].max()
            inc_start = (date.fromisoformat(last) - timedelta(days=7)).isoformat()
            raw = _with_retries(self.fetch_history, symbol, inc_start, end, retries=self.retries,
                                what=f"{symbol} prices (incremental)")
            self.network_calls += 1
            new = clean_prices(raw)
            combined = pd.concat([cached[cached["date"] < inc_start], new], ignore_index=True)
            combined = combined.drop_duplicates("date", keep="last").sort_values("date")
            # recompute returns across the seam
            combined["ret"] = combined["adj_close"].pct_change()
            combined["log_ret"] = np.log(combined["adj_close"]).diff()
            status = "incremental"
        else:
            raw = _with_retries(self.fetch_history, symbol, start, end, retries=self.retries,
                                what=f"{symbol} prices")
            self.network_calls += 1
            combined = clean_prices(raw)
            status = "downloaded"
        if combined.empty:
            raise ValueError(f"No price rows returned for {symbol}")
        combined.to_csv(_prices_path(self.cache, symbol), index=False)
        _meta_path(self.cache, symbol).write_text(json.dumps({
            "symbol": symbol, "start": min(start, meta.get("start", start)), "end": end,
            "pulled_at": datetime.now(timezone.utc).isoformat(), "rows": len(combined)}))
        return combined, status

    def info(self, symbol: str) -> dict:
        p = self.cache / f"{symbol}_info.json"
        if p.exists() and (time.time() - p.stat().st_mtime) < float(self.m.get("info_cache_days", 30)) * 86400:
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
        try:
            info = _with_retries(self.fetch_info, symbol, retries=self.retries, what=f"{symbol} info")
            self.network_calls += 1
        except Exception as exc:
            log.warning("%s: company info unavailable (%s)", symbol, exc)
            return {}
        keep = {k: info.get(k) for k in ("longName", "shortName", "sector", "industry", "marketCap",
                                         "sharesOutstanding", "quoteType", "exchange", "currency", "country")}
        p.write_text(json.dumps(keep, default=str))
        return keep

    def earnings_dates(self, symbol: str) -> pd.DataFrame:
        p = self.cache / f"{symbol}_earnings_dates.csv"
        if p.exists() and (time.time() - p.stat().st_mtime) < float(self.m.get("info_cache_days", 30)) * 86400:
            return pd.read_csv(p)
        try:
            raw = _with_retries(self.fetch_earnings_dates, symbol, int(self.m.get("earnings_dates_limit", 100)),
                                retries=self.retries, what=f"{symbol} earnings dates")
            self.network_calls += 1
        except Exception as exc:
            log.warning("%s: Yahoo earnings dates unavailable (%s); EPS surprise will be blank", symbol, exc)
            return pd.DataFrame()
        df = clean_earnings_dates(raw)
        df.to_csv(p, index=False)
        return df


def pull_market_data(cfg: Config, con: sqlite3.Connection, puller: MarketDataPuller | None = None) -> dict[str, Any]:
    puller = puller or MarketDataPuller(cfg)
    overrides = cfg.market.get("symbol_overrides") or {}
    pause = float(cfg.market.get("pause_between_tickers_sec", 1.0))
    benchmarks = [str(b).upper() for b in (cfg.market.get("benchmarks") or ["SPY"])]
    stats: dict[str, Any] = {"ok": 0, "cached": 0, "failed": []}

    targets = [(t, False) for t in cfg.tickers] + [(b, True) for b in benchmarks if b not in cfg.tickers]
    for ticker, is_bench in targets:
        sym = yf_symbol(ticker, overrides)
        before = puller.network_calls
        try:
            prices, status = puller.prices(sym)
        except Exception as exc:
            log.error("%s (%s): price download failed - %s", ticker, sym, exc)
            stats["failed"].append(ticker)
            if not is_bench:
                db.update_partial(con, "companies", "ticker", [{"ticker": ticker, "yf_symbol": sym,
                                                                "yf_status": f"failed: {str(exc)[:200]}"}])
            con.commit()
            continue
        rows = prices.assign(ticker=ticker).to_dict("records")
        db.replace_where(con, "market_data", "ticker=?", (ticker,), rows)
        stats["cached" if status == "cached" else "ok"] += 1

        if not is_bench:
            info = puller.info(sym)
            ed = puller.earnings_dates(sym)
            if len(ed):
                db.replace_where(con, "yf_earnings_dates", "ticker=?", (ticker,),
                                 ed.assign(ticker=ticker).to_dict("records"))
            db.update_partial(con, "companies", "ticker", [{
                "ticker": ticker, "yf_symbol": sym,
                "yf_sector": info.get("sector"), "yf_industry": info.get("industry"),
                "yf_market_cap": info.get("marketCap"), "yf_shares_out": info.get("sharesOutstanding"),
                "yf_quote_type": info.get("quoteType"),
                "yf_status": f"ok ({status}, {len(prices)} rows, {prices['date'].min()}..{prices['date'].max()})",
            }])
            # don't overwrite the transcript company name with NULL
            con.execute("UPDATE companies SET company_name = COALESCE(company_name, ?) WHERE ticker=?",
                        (info.get("longName") or info.get("shortName"), ticker))
        con.commit()
        log.info("%-6s %-11s %5d rows %s..%s", ticker, status, len(prices), prices["date"].min(), prices["date"].max())
        if puller.network_calls > before:
            time.sleep(pause)
    log.info("Market data finished: %d downloaded/updated, %d from cache, %d failed %s",
             stats["ok"], stats["cached"], len(stats["failed"]), stats["failed"])
    return stats
