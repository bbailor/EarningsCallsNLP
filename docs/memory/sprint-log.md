# Sprint log

One block per sprint, newest first. Add evidence links as they exist.

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
