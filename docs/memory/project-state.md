# Project state

Last updated: 2026-11-11 (end of Sprint 2 report). Update this file whenever a work unit changes status.

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
| NLP-03 | Unity cluster compatibility | Written, NOT tested on Unity |
| NLP-04 | Transcript source documentation | [status ____] |
| NLP-05 | Top-10 S&P 500, five-year pull (about 200 calls) | Planned for Sprint 3 |
| NLP-06 | Transcript-to-NLP-variable workflow | Pretrained FinBERT + BERTopic stages built and run (Sprint 2): 163 calls, 9 tickers, 54 topics, 69,546 scored sentences. Full five-year sample not run yet |
| NLP-07 | Baseline NLP variables and summary statistics | Planned for Sprint 3 |
| (none) | Hand-labeled test set (300-500 mentions) | Planned for Sprint 3 |
| (none) | First version of our own model | Planned for Sprint 3 |
| (none) | Streamlit dashboard | Later; pipeline already stores what it needs |

## Open questions
- Exactly what counts as "our own model" for Prof. Glatzer (baseline classifier, fine-tuned DistilBERT, or both)?
- Unity account and project storage path.
- Who on the team labels the test set, and the labeling guidelines.
- Purchases under consideration: Seeking Alpha (POT-05), Capital IQ premium (POT-06), LLM API access (POT-04).
