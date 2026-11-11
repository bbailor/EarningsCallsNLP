---
paths:
  - "src/ecnlp/**/*.py"
  - "config.yaml"
  - "tests/**/*.py"
---

# Data pipeline rules

## earningscalls.dev
- Ultra plan: 20,000 requests a month. Failed lookups count too.
- Cache every response in `data/raw/earningscalls/`. A second run must make almost no requests.
- Two kinds of HTTP 429: a rate limit (wait and retry) and quota exhausted (stop cleanly and keep progress). Never treat them the same.
- Response format is not fully documented. Parse defensively and confirm against `pipeline check` output.
- Look up up to 100 tickers per search request. Limit results to US earnings calls.
- GOOG and GOOGL share calls, so remove duplicates. Convert `BRK.B` to `BRK-B` for Yahoo.

## Yahoo Finance (yfinance)
- yfinance returns adjusted prices only by default. Request adjusted and raw prices explicitly.
- Yahoo rate-limits. Retry with backoff, and log delisted tickers instead of crashing.

## Call timing and event windows
- Every call gets `call_session`: `before_open`, `market_hours` or `after_close`.
- `day0` is the first trading day the market could react. An after-close call reacts the next trading day.
- Calls with no time listed are flagged, never assigned a guessed day.
- Timestamps are converted to US/Eastern. If NVDA calls aren't about 4:30-5 pm ET, fix `naive_timestamp_tz`.
- Returns: raw, minus S&P 500, and market-model abnormal returns, plus abnormal volume and EPS surprise.

## Cleaning
- Remove operator lines and legal boilerplate. Label each sentence prepared remarks or Q&A.
- The sentence splitter must handle "Mr.", "U.S." and "$4.5 billion".
- `sentence_id` must stay the same across re-runs.

## Tests
- Every new function gets a pytest test using mocked APIs. Never call live APIs in tests.
