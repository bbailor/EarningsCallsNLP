# Earnings Call Technology Analyzer — Data + NLP Pipeline

Pulls earnings call transcripts that mention AI from **earningscalls.dev**, pulls matching market data from **Yahoo Finance (yfinance)**, and cleans/organizes both into a SQLite database plus analysis-ready files for the FinBERT/BERTopic stage. Covers work units **NLP-01, NLP-02, NLP-03**, and lays the groundwork for **POT-02** (keyword dictionary) and **POT-03** (event-window returns).

## Quick start

```bash
conda env create -f environment.yml && conda activate ecnlp   # or: pip install -e . -r requirements.txt
cp .env.example .env            # paste your Ultra API key into .env
python -m ecnlp.pipeline check  # ~4 requests: verifies key, quota, live response schema
python -m ecnlp.pipeline all    # transcripts -> market -> prepare -> events -> export
pytest                          # offline tests (mocked API, yfinance, FinBERT and BERTopic)
```

Run stages individually with `transcripts`, `market`, `prepare`, `events`, `export`. Override the ticker list for a quick test: `--tickers NVDA,MSFT`. Everything is configured in `config.yaml`.

## What each stage does

| Stage | Source | Output |
|---|---|---|
| `transcripts` | earningscalls.dev | `companies`, `calls`, `segments` (one row per speaker turn) |
| `market` | yfinance | `market_data` (daily OHLCV + adj close + returns), `yf_earnings_dates` (EPS estimate/actual/surprise), company info on `companies` |
| `prepare` | offline | `segments.text_clean`, `sentences` (stable IDs, section, speaker role, safe‑harbor flag, AI-keyword flag) |
| `events` | offline | `event_returns`: day 0, market‑adjusted and market‑model CARs per window, abnormal volume, EPS surprise, quality flags |
| `export` | offline | `data/processed/` files below |

**Exports** (`data/processed/`):
- `nlp_sentences.parquet` — every non‑boilerplate, non‑operator sentence with call/company metadata. Input for FinBERT.
- `nlp_ai_contexts.parquet` — each AI‑term sentence with ±1 sentence of context from the same speaker (the SDS mention unit). Input for FinBERT + BERTopic.
- `nlp_segments.parquet` — full speaker turns, if you prefer turn‑level modeling.
- `calls_master.csv` — one row per call: transcript stats, AI sentence density, sector, market cap, and every event‑window return. NLP scores get joined here on `call_id` for the cross‑reference.
- `coverage_report.csv` — per ticker: calls in API, calls mentioning AI, calls with text, Yahoo status, events with complete windows (documents sample attrition, per SDS §3.1).

## NLP workflow (Sprint 2): FinBERT + BERTopic

Sample: the 10 largest S&P 500 companies (`tickers` in `config.yaml`), calls from `date_from: 2021-11-01` (last five years), **all** earnings calls (`include_non_ai_calls: true`, `calls.mentions_ai` keeps the AI flag). Yahoo Finance is not needed for this workflow; skip the `market`/`events` stages.

```bash
pip install -e ".[nlp]"                       # torch, transformers, bertopic, ...
python -m ecnlp.pipeline transcripts          # pull (cached; ~1.2 requests per call)
python -m ecnlp.pipeline prepare              # clean + sentence split
python -m ecnlp.pipeline nlp                  # sentiment -> topics -> scores
# or on Unity: sbatch slurm/pull_data.sbatch, then sbatch slurm/nlp.sbatch (GPU)
```

| Stage | What it does | Output |
|---|---|---|
| `sentiment` | FinBERT (`ProsusAI/finbert`) on every non-boilerplate sentence and on every AI mention with ±1 sentence of context. Incremental: only new or changed text is scored, committed every 2,000 sentences. | `sentiment` table (p_positive / p_negative / p_neutral, label, score = p_pos − p_neg) |
| `topics` | One BERTopic model (MiniLM embeddings → UMAP → HDBSCAN → c-TF-IDF, earnings-call filler removed) over all sentences in the sample, so topic ids are comparable across firms and quarters. Refit every run; embeddings are cached in `data/models/`. | `topics`, `topic_assignments`; saved model in `data/models/bertopic_sentences/` |
| `scores` | Aggregates both to one row per call. | `call_scores` table + files below; also joined into `calls_master.csv` |

**NLP variables per call** (`data/processed/nlp_call_scores.csv`, one row per call with ticker / fiscal period / date):

- `sent_<slice>_{n, mean, net_tone, pct_pos, pct_neg}` for slices `all`, `prepared` (prepared remarks), `qa`, `exec`, `analyst`, `nonai` (sentences without AI terms) and `ai_ctx` (AI mentions with context). `mean` averages p_pos − p_neg; `net_tone` = (#pos − #neg) / (#pos + #neg).
- `sent_ai_minus_nonai`: how much more positive management is about AI than about everything else.
- `topic_entropy` (0 = one topic, 1 = evenly spread), `topic_dominant`, `topic_outlier_share`, `topic_<id>_share` for the 20 largest topics (names in `nlp_topics.csv`).

Other exports: `nlp_topic_shares.csv` (every call × topic share, long format), `nlp_sentence_scores.parquet` (sentence-level label, probabilities, topic) and `nlp_topics.csv` (topic words + representative sentence).

Knobs in `config.yaml` under `nlp:`: switch FinBERT to `yiyanghkust/finbert-tone`, model AI themes only with `topics.corpus: ai_contexts`, change topic granularity with `min_topic_size` / `nr_topics`. Changing the sentiment model scores everything again under the new model name; the old scores are kept.

Caveats to discuss as a team:
- FinBERT reads direction words, not accounting meaning. "Past dues were down 31 basis points" is scored negative. Spot-check `nlp_sentence_scores.parquet` before relying on small differences.
- BERTopic is stochastic across machines even with `seed`. Fit the final model once and keep `data/models/` with the results.
- The NLP stages only cover the configured tickers and date range, so leftover pilot calls in the database (e.g. CAT) are excluded.
- Section split: Q&A starts at the first analyst turn **after an executive has spoken**. Without that guard, IR heads the API tags as `analyst` (NVIDIA) made whole calls "Q&A".

## How the earningscalls.dev pull works (and stays inside the Ultra quota)

Ultra = **60 req/min, 20,000 req/month**, and 400/404 responses still count. The pipeline therefore:

1. **Discovery** — `/search/by_ticker` with the AI query returns matching call IDs for up to 100 tickers in **one** request. If an ID list comes back truncated, it falls back to paging `/search/` for that ticker.
2. **Call lists** — `/earnings/?ticker=X&country=US&event_type=earnings` gives dates, fiscal period and company metadata. Intersecting with step 1 removes non‑earnings events and foreign companies that share a symbol (e.g. PNB).
3. **Text** — `/speakers/{id}` (200 turns/page, usually 1 request per call). Older transcripts without speaker IDs fall back to `/transcripts/{id}`.

Before fetching, it reads your usage from `/me`, estimates the requests needed, and stops cleanly before the monthly safety margin. A hard per‑run cap (`max_requests_per_run`) is also enforced. Everything is cached under `data/raw/earningscalls/`, so **a re‑run makes zero transcript requests** and an interrupted run resumes where it stopped. Rough cost: ~1.2 requests per call + 1 per ticker, so the full S&P 500 over 2020–2026 (~10–13k AI-mentioning calls, ~13–16k requests) fits in one month across a few runs of the 6,000-request cap.

## Issues found and handled before writing the code

- **Auth/endpoint details** were verified against the official OpenAPI spec: header `X-API-Key`, base `https://earningscalls.dev/api/v1`. If your Ultra plan was bought through **RapidAPI**, set `earningscalls.provider: rapidapi` (different host and headers).
- **429 means two different things** (per‑minute vs monthly quota). Per‑minute → wait and retry; monthly → stop and save progress.
- **Ticker collisions** across exchanges → US filter + issuer de‑duplication; share classes (GOOG/GOOGL) de‑duplicated by call ID with a warning.
- **Call timing**: an after‑close call (5 pm ET) can only move the stock the next trading day. Each call is classified `pre_market` / `intraday` / `after_close` in ET and day 0 is set accordingly. Date‑only or midnight‑UTC timestamps are **not** shifted to the prior evening; they're flagged `session_unknown`.
- **yfinance changed its default to `auto_adjust=True`**, which silently drops raw Close. We request both explicitly. Rate‑limit errors retry with backoff; delisted tickers are logged in `companies.yf_status` instead of crashing the run. `BRK.B` → `BRK-B` symbol mapping is automatic (`symbol_overrides` for anything else).
- **Missing fiscal year/quarter** in the API is filled from the call title (e.g. "Q1 2025").
- **Q&A vs prepared remarks** is derived locally with the same rule the API uses (first analyst turn), saving a request per call.

## Remaining uncertainties (verify with `check` on the first run)

The API docs don't publish response bodies for `/speakers`, `/transcripts` or `/me`. The code parses these defensively using the field names from the official examples (`text_content`, `component_order`, `speaker_name`, `speaker_type`). `python -m ecnlp.pipeline check` prints the live keys and three sample segments so you can confirm. It also prints recent NVDA call times: they should read ~4:30–5 pm ET. If they're off by 4–5 hours, change `naive_timestamp_tz` in `config.yaml`.

## Design choices to confirm as a team

- **Control group.** The spec pulls only calls that mention AI. For the regressions you'll likely want non‑AI calls too; set `include_non_ai_calls: true` and every earnings call is kept with `calls.mentions_ai = 0/1` (costs more requests).
- **AI keyword list** in `config.yaml` is preliminary (POT‑02). `prepare` warns if an AI‑matched call contains none of your keywords, which points to dictionary gaps.
- **Capital IQ → yfinance.** The SDS still references Capital IQ; `market_data` keeps the same table name so the SDS ERD only needs the source updated. The new `segments` and `yf_earnings_dates` tables should be added to the ERD.

## Unity cluster (NLP‑03)

Edit `ECNLP_DATA_DIR` in `slurm/pull_data.sbatch` to your project storage path, put the key in `.env` (`chmod 600 .env`), then `sbatch slurm/pull_data.sbatch`. Re‑submitting resumes from cache.

## Layout

```
config.yaml                 all settings
src/ecnlp/
  earningscalls_client.py   throttled, cached, quota-aware HTTP client
  transcripts.py            NLP-01 discovery + pull
  market_data.py            NLP-02 yfinance pull + cleaning
  cleaning.py               normalization, sections, sentence splitter, AI-term matcher
  prepare.py                segments -> sentences
  timeutils.py / events.py  call timing, day 0, event-window returns
  exports.py                parquet/csv exports + coverage report
  sentiment.py              FinBERT sentence / AI-context sentiment
  topics.py                 BERTopic topic model
  nlp_scores.py             call-level NLP variables + nlp_* exports
  db.py                     SQLite schema (SDS §5)
  pipeline.py               CLI
tests/                      mocked end-to-end tests
slurm/pull_data.sbatch      Unity job script (transcript pull)
slurm/nlp.sbatch            Unity GPU job script (NLP stages)
```
