"""Configuration loading.

Settings come from config.yaml, with two environment overrides so the same
code runs on a laptop and on the Unity cluster:

* ECNLP_CONFIG    - path to the YAML file (default: ./config.yaml)
* ECNLP_DATA_DIR  - replaces `data_dir` from the YAML
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

try:  # optional: lets the team keep the API key in a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class ConfigError(ValueError):
    """Raised when config.yaml is missing or invalid."""


@dataclass
class Paths:
    root: Path

    @property
    def raw_ec(self) -> Path:
        return self.root / "raw" / "earningscalls"

    @property
    def raw_yf(self) -> Path:
        return self.root / "raw" / "yfinance"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def db(self) -> Path:
        return self.root / "earnings_nlp.sqlite"

    def ensure(self) -> None:
        for p in (self.raw_ec, self.raw_yf, self.processed, self.logs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    raw: dict[str, Any]
    paths: Paths
    tickers: list[str]
    date_from: str
    date_to: str
    ec: dict[str, Any] = field(default_factory=dict)
    market: dict[str, Any] = field(default_factory=dict)
    cleaning: dict[str, Any] = field(default_factory=dict)
    events: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.ec.get("api_key_env", "EARNINGSCALLS_API_KEY"))


def _norm_ticker(t: Any) -> str:
    return str(t).strip().upper()


def _validate_date(s: str, name: str) -> str:
    try:
        date.fromisoformat(s)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be YYYY-MM-DD, got {s!r}") from exc
    return s


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    path = Path(path or os.environ.get("ECNLP_CONFIG", "config.yaml"))
    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        raw = copy.deepcopy(raw)
        raw.update(overrides)

    data_dir = os.environ.get("ECNLP_DATA_DIR") or raw.get("data_dir") or "./data"
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = (path.parent / data_path).resolve()
    paths = Paths(data_path)

    tickers = [_norm_ticker(t) for t in (raw.get("tickers") or [])]
    csv_path = raw.get("tickers_csv")
    if csv_path:
        import pandas as pd

        csv_file = Path(csv_path)
        if not csv_file.is_absolute():
            csv_file = path.parent / csv_file
        df = pd.read_csv(csv_file)
        col = next((c for c in df.columns if c.lower() in ("ticker", "symbol")), None)
        if col is None:
            raise ConfigError(f"{csv_file} needs a 'ticker' or 'symbol' column")
        tickers += [_norm_ticker(t) for t in df[col].dropna()]
    # de-duplicate, keep order
    tickers = list(dict.fromkeys(t for t in tickers if t))
    if not tickers:
        raise ConfigError("No tickers configured (tickers / tickers_csv)")

    date_from = _validate_date(str(raw.get("date_from") or "2020-01-01"), "date_from")
    date_to = _validate_date(str(raw.get("date_to") or date.today().isoformat()), "date_to")
    if date_from > date_to:
        raise ConfigError("date_from is after date_to")

    ec = raw.get("earningscalls") or {}
    if ec.get("provider", "direct") not in ("direct", "rapidapi"):
        raise ConfigError("earningscalls.provider must be 'direct' or 'rapidapi'")

    return Config(
        raw=raw,
        paths=paths,
        tickers=tickers,
        date_from=date_from,
        date_to=date_to,
        ec=ec,
        market=raw.get("market") or {},
        cleaning=raw.get("cleaning") or {},
        events=raw.get("events") or {},
    )
