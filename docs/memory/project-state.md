# Project state

Last updated: 2026-11-18 (Sprint 3, returns for all tickers). Update this file whenever a work unit changes status.

## Team and course
- Team GPT Gains. Ben Bailor owns the data pipeline and NLP setup. Advisor: Prof. Glatzer.
- Sprints run about one to two weeks. Status reports are individual. Work plans use NLP-xx (necessary) and POT-xx (potential) work unit IDs.

## Data sources
- Transcripts: earningscalls.dev, Ultra plan (20,000 requests/month).
- Market data: Yahoo Finance through yfinance (prices, company info, EPS surprise).
- Capital IQ is not used. Mass export needs the premium plug-in we don't have (POT-06).

## Work unit status
| ID | Work unit | Status |
| --- | --- | --- |
| NLP-01 | earningscalls.dev transcript pull | Done (Sprint 1) |
| NLP-02 | Yahoo Finance data pull | Done (Sprint 1) |
| (none) | Transcript cleaning and sentence preparation | Done (Sprint 1) |
| POT-03 | Event-window returns, first version | Done (Sprint 1) |
| (none) | 16 automated tests, usage guide, private GitHub repo | Done (Sprint 1); 22 tests after the NLP stages (Sprint 2) |
| POT-02 | AI keyword dictionary in config.yaml | Done (Sprint 2) |
| NLP-08 | Model comparison and evaluation plan | Plan written (Sprint 2); comparison not run yet |
| NLP-03 | Cloud compute compatibility (AWS; was Unity until 2026-11-18) | Switched to AWS 2026-11-18. AWS setup not started. Unity `slurm/` scripts were written but never tested |
| NLP-04 | Transcript source documentation | [status ____] |
| NLP-05 | Top-10 S&P 500, five-year pull (about 200 calls) | Transcripts: 163 calls, 9 tickers, 2021-11-17 to 2026-11-02 (AI-mentioning calls only); BRK.B has 0 calls in the API. Market, events, export re-run for all tickers 2026-11-18: 166 of 166 calls have `car_mkt_0_1`. Open: BRK.B and CAT decisions |
| NLP-06 | Transcript-to-NLP-variable workflow | Re-run on the five-year sample 2026-11-18: 163 calls, 9 tickers, 56 topics (was 54 on 2026-11-11, cause [____]), 69,546 scored sentences, 60 call-level variables; joined into `calls_master.csv`. Version logging added 2026-11-18 (Python, packages, HF model revisions in the log and `run_log.notes`). Open: the 2026-11-18 `nlp` run was launched with system Python 3.12, not `.venv`, so its package versions are unknown; re-run `nlp` in `.venv` for a versioned log. CAT (3 calls) not scored |
| NLP-07 | Baseline NLP variables and summary statistics | First version done 2026-11-18: `pipeline summary` writes 4 tables and 4 figures to `data/processed/summary/` (163 calls, 9 tickers). Correlations with returns are descriptive only (no clustering by company) |
| (none) | Hand-labeled test set (300-500 mentions) | Planned for Sprint 3 |
| (none) | First version of our own model | Planned for Sprint 3 |
| (none) | Streamlit dashboard | Later; pipeline already stores what it needs |

## Known data gaps (checked 2026-11-11)
- Fixed 2026-11-18: `market`, `events`, `export` re-run for all tickers; every call has `car_mkt_0_1`. `events_all_windows` is one short of the call count for AAPL, GOOGL, JPM, META, MSFT, TSLA (reason [____]).
- NLP outputs re-run 2026-11-18 and cover 163 of 166 calls; the 3 unscored are CAT (not in config tickers).
- `include_non_ai_calls` is `false` in config.yaml, but decisions.md (2026-11-11) says `true`. Confirm which is current before the next pull; `true` costs extra requests.
- CAT (3 calls) is left over from the test run and is not in the top-10 ticker list. Decide whether to drop it from analysis.
- BRK.B: Berkshire holds no quarterly calls, so the sample is effectively 9 companies. Decide whether to replace it (LLY or WMT).

## Open questions
- Exactly what counts as "our own model" for Prof. Glatzer (baseline classifier, fine-tuned DistilBERT, or both)?
- AWS: which service (EC2 GPU, SageMaker), whose account, budget, and where data and the API key live (not in the repo or a shared bucket).
- Who on the team labels the test set, and the labeling guidelines.
- Purchases under consideration: Seeking Alpha (POT-05), Capital IQ premium (POT-06), LLM API access (POT-04).
