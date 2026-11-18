# Sprint log

One block per sprint, newest first. Add evidence links as they exist.

## Sprint 3 (in progress)
- 2026-11-18: NLP-05 returns: `market`, `events`, `export` re-run for all tickers (Individual). Evidence: `data/processed/calls_master.csv` (166 of 166 calls have `car_mkt_0_1`), `coverage_report.csv`, logs `market_20261118_101716.log`, `events_20261118_101817.log`, `export_20261118_101827.log`.
- 2026-11-18: NLP-06 re-run on the full sample (Individual). Sentiment: 76,743 units already scored with ProsusAI/finbert, 0 new. BERTopic re-fit on 60,094 sentences: 56 topics. Call scores: 163 calls x 60 variables; export re-run so `calls_master.csv` carries them. Evidence: log `nlp_20261118_102027.log`, `data/processed/nlp_call_scores.csv`, `nlp_topics.csv`.
- 2026-11-18: NLP-06 version logging (`src/ecnlp/runinfo.py`) and numba debug noise removed from logs (Individual). Evidence: `scores_20261118_103305.log` (12 lines; torch 2.15.0, transformers 5.22.1, bertopic 0.17.4, ProsusAI/finbert @ 0bc2ac10, all-MiniLM-L6-v2 @ 1110a243). Commit [____].
- 2026-11-18: NLP-07 first version (Individual). `src/ecnlp/summary_stats.py`, `pipeline summary` stage. Outputs in `data/processed/summary/`: `summary_overall.csv`, `summary_by_ticker.csv`, `summary_by_year.csv`, `correlations_with_car.csv`, fig1-fig4 PNGs. Pytest: 26 passed in 5.5s. Commit [____].

## Sprint 2 (report 2026-11-11)
- Completed: model and evaluation plan (NLP-08 partial); AI keyword dictionary (POT-02); comparison group and SDS fixes; pretrained FinBERT/BERTopic stages built and run on the current sample (NLP-06, Individual); topic entropy scaling fix by Kaab Dawit (Team); 6 NLP tests added (22 total); tests and repo upkeep.
- Written but untested: Unity sbatch script and environment.yml (NLP-03).
- Next: five-year top-10 pull (NLP-05); pretrained models on full sample plus summary statistics (NLP-06/07); one-page model write-ups; hand-labeled test set; first own-model baseline; Unity test job.
- Evidence:
  - Commits: `b5c56e5` FinBERT and BERTopic NLP stages (PR #2, 2026-11-09); `ae69a3e` topic entropy scaling and `084ce4f` .gitignore fix (PR #3, 2026-11-11). Repo: github.com/bbailor/EarningsCallsNLP
  - Output files (2026-11-11): `data/processed/nlp_call_scores.csv` (163 calls, 9 tickers), `nlp_topics.csv` (54 topics), `nlp_topic_shares.csv`, `nlp_sentence_scores.parquet` (69,546 sentences)
  - Pytest: 22 passed in 3.9s (run 2026-11-11 in `.venv`)

## Sprint 1 (report 2026-10-28)
- Completed: NLP-01 transcript pull; NLP-02 Yahoo pull; cleaning and sentence prep; POT-03 event returns; 16 tests; usage guide; private repo on master.
- Professor feedback: show concrete evidence; separate individual and team work; make goals measurable; start with pretrained models and learn their structure; list Unity as pending.
