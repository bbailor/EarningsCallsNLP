"""HTTP client for the earningscalls.dev REST API.

Facts taken from the official OpenAPI spec (https://earningscalls.dev/openapi.json)
and docs (https://earningscalls.dev/docs):

* Base URL  https://earningscalls.dev/api/v1, auth header ``X-API-Key``.
  RapidAPI subscribers use https://earnings-call-transcripts1.p.rapidapi.com/api/v1
  with ``X-RapidAPI-Key`` / ``X-RapidAPI-Host`` instead.
* Ultra tier: 60 requests/minute, 20,000/month.
* Every request that reaches the API counts toward quota, *including 400/404*.
  401/403/429 do not count.
* 429 = quota exceeded; the body includes the reset date.

The client adds: a sliding-window throttle, retries with backoff on 5xx and
network errors, a request counter with a hard per-run cap, and a JSON disk
cache so re-runs cost zero requests for data already pulled.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DIRECT_BASE = "https://earningscalls.dev/api/v1"
RAPID_HOST = "earnings-call-transcripts1.p.rapidapi.com"
RAPID_BASE = f"https://{RAPID_HOST}/api/v1"


class EarningsCallsError(RuntimeError):
    """Base error for the API client."""


class AuthError(EarningsCallsError):
    """401 - missing or invalid key."""


class TierError(EarningsCallsError):
    """403 - endpoint not included in the subscription tier."""


class QuotaExceeded(EarningsCallsError):
    """429 monthly quota, or the per-run request cap configured locally."""


class EarningsCallsClient:
    def __init__(
        self,
        api_key: str,
        provider: str = "direct",
        cache_dir: Path | None = None,
        requests_per_minute: int = 55,
        max_requests: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise AuthError(
                "No earningscalls.dev API key. Set the EARNINGSCALLS_API_KEY environment "
                "variable (or put it in a .env file next to config.yaml)."
            )
        self.provider = provider
        if provider == "rapidapi":
            self.base = RAPID_BASE
            self.headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": RAPID_HOST}
        else:
            self.base = DIRECT_BASE
            self.headers = {"X-API-Key": api_key}
        self.headers["Accept"] = "application/json"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rpm = max(1, int(requests_per_minute))
        self.max_requests = max_requests
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._stamps: deque[float] = deque()
        self.requests_made = 0  # requests that reached the API this run
        self.cache_hits = 0

    # ------------------------------------------------------------------ cache
    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        if len(safe) > 120:  # keep file names short but unique
            safe = safe[:80] + "_" + hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe}.json"

    def _cache_read(self, key: str, max_age_days: float | None) -> Any | None:
        p = self._cache_path(key)
        if p is None or not p.exists():
            return None
        if max_age_days is not None and (time.time() - p.stat().st_mtime) > max_age_days * 86400:
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Corrupt cache file %s - refetching", p)
            return None
        self.cache_hits += 1
        return payload

    def _cache_write(self, key: str, payload: Any) -> None:
        p = self._cache_path(key)
        if p is None:
            return
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)  # atomic: a crash never leaves a half-written cache file

    # --------------------------------------------------------------- throttle
    def _throttle(self) -> None:
        now = time.monotonic()
        while self._stamps and now - self._stamps[0] >= 60:
            self._stamps.popleft()
        if len(self._stamps) >= self.rpm:
            wait = 60 - (now - self._stamps[0]) + 0.05
            log.debug("Rate limit window full; sleeping %.1fs", wait)
            time.sleep(max(wait, 0))
        self._stamps.append(time.monotonic())

    # ---------------------------------------------------------------- request
    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        cache_key: str | None = None,
        max_age_days: float | None = None,
        allow_404: bool = True,
    ) -> Any | None:
        """GET ``path`` (relative to /api/v1). Returns parsed JSON, or None on 404."""
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        if cache_key:
            cached = self._cache_read(cache_key, max_age_days)
            if cached is not None:
                return cached.get("__payload__", cached) if isinstance(cached, dict) else cached

        url = f"{self.base}/{path.lstrip('/')}"
        attempt = 0
        while True:
            if self.max_requests is not None and self.requests_made >= self.max_requests:
                raise QuotaExceeded(
                    f"Per-run request cap reached ({self.max_requests}). Re-run to continue; "
                    "cached data will be skipped."
                )
            self._throttle()
            try:
                resp = self.session.get(url, headers=self.headers, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise EarningsCallsError(f"Network error on {url}: {exc}") from exc
                backoff = min(2**attempt, 60)
                log.warning("Network error (%s); retry %d in %ss", exc, attempt, backoff)
                time.sleep(backoff)
                continue

            status = resp.status_code
            if status not in (401, 403, 429):
                self.requests_made += 1
            log.debug("GET %s %s -> %s", path, params, status)

            if status == 200:
                try:
                    payload = resp.json()
                except ValueError as exc:
                    raise EarningsCallsError(f"Non-JSON response from {url}") from exc
                if cache_key:
                    self._cache_write(cache_key, {"__payload__": payload})
                return payload
            msg = _error_message(resp)
            if status == 404 and allow_404:
                log.info("404 for %s (%s)", path, msg)
                return None
            if status == 401:
                raise AuthError(f"401 Unauthorized - check the API key ({msg})")
            if status == 403:
                raise TierError(f"403 Forbidden on {path} - tier does not include this endpoint ({msg})")
            if status == 429:
                retry_after = resp.headers.get("Retry-After")
                low = msg.lower()
                monthly = any(w in low for w in ("month", "quota", "reset")) and not retry_after
                if monthly:
                    raise QuotaExceeded(f"Monthly quota exceeded: {msg}")
                attempt += 1
                if attempt > self.max_retries:
                    raise QuotaExceeded(f"Still rate limited after {self.max_retries} retries: {msg}")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 61
                log.warning("429 rate limited; sleeping %ss", wait)
                time.sleep(wait)
                continue
            if status >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise EarningsCallsError(f"{status} from {url} after retries: {msg}")
                backoff = min(2**attempt, 60)
                log.warning("%s server error on %s; retry %d in %ss", status, path, attempt, backoff)
                time.sleep(backoff)
                continue
            raise EarningsCallsError(f"{status} from {url}: {msg}")

    # ------------------------------------------------------------- endpoints
    def me(self) -> dict | None:
        return self.get("me")

    def list_earnings(self, **params) -> dict | None:
        return self.get("earnings/", params=params)

    def search_by_ticker(self, **params) -> dict | None:
        return self.get("search/by_ticker", params=params)

    def search(self, **params) -> dict | None:
        return self.get("search/", params=params)

    def speakers(self, earnings_id: int, page: int = 1, limit: int = 200) -> dict | None:
        return self.get(
            f"speakers/{earnings_id}",
            params={"page": page, "limit": limit},
            cache_key=f"speakers_{earnings_id}_p{page}_l{limit}",
        )

    def transcript(self, earnings_id: int, offset: int = 0, limit: int = 100000) -> dict | None:
        return self.get(
            f"transcripts/{earnings_id}",
            params={"offset": offset, "limit": limit},
            cache_key=f"transcript_{earnings_id}_o{offset}",
        )


def _error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            return str(body.get("message") or body.get("error") or body)
        return str(body)
    except ValueError:
        return resp.text[:300]


def parse_usage(me_payload: Any) -> dict[str, int | None]:
    """Pull (used, limit, remaining) out of the /me response.

    The response schema is not documented, so we search the JSON for the
    obvious key names instead of hard-coding a path.
    """
    found: dict[str, int | None] = {"used": None, "limit": None, "remaining": None}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if "remaining" in kl and found["remaining"] is None:
                        found["remaining"] = int(v)
                    elif ("used" in kl or kl in ("usage", "requests_this_month", "count")) and found["used"] is None:
                        found["used"] = int(v)
                    elif "limit" in kl and "minute" not in kl and found["limit"] is None:
                        found["limit"] = int(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(me_payload)
    if found["remaining"] is None and found["used"] is not None and found["limit"] is not None:
        found["remaining"] = found["limit"] - found["used"]
    return found
