# Sprint 3 Work Plan — Team GPT Gains (DRAFT)

Sprint dates: [start ____] to [end ____]
Status: draft from `docs/memory/project-state.md` on 2026-11-11. Confirm IDs, owners and priorities as a team before copying into the Word plan.

## Necessary work units

| ID | Name | Priority | Description with a "Done when" | Status | Owner |
| --- | --- | --- | --- | --- | --- |
| NLP-05 | Top-10 S&P 500, five-year sample | High | Finish the sample: market data and event returns for every ticker, and a decision on BRK.B and CAT. Done when `coverage_report.csv` shows `events_all_windows` > 0 for every kept ticker and `calls_master.csv` has `car_mkt_0_1` for at least [____]% of calls (currently 6 of 166). Blocker: Yahoo rate limits; team decision on BRK.B (replace with LLY or WMT?) and CAT. | Transcripts pulled (163 calls, 9 tickers); returns missing | Ben Bailor |
| NLP-06 | Transcript-to-NLP-variable workflow | High | Run FinBERT and BERTopic on the final sample. Done when `nlp_call_scores.csv` has one row per call in `calls_master.csv` and the run log records the model names and versions. Blocker: NLP-05 must finish first. | Built and run on the current sample | Ben Bailor |
| NLP-07 | Baseline NLP variables and summary statistics | High | Summary statistics for FinBERT sentiment and word counts by company and year. Done when a table and at least [____] figures are saved under `data/processed/` or `docs/` and cited in the status report. Blocker: needs NLP-05 returns and the NLP-06 rerun. | Not started | [____] |
| NLP-03 | Unity cluster compatibility | Medium | Run one test job on Unity. Done when `slurm/pull_data.sbatch` or `slurm/nlp.sbatch` finishes on Unity and the `slurm-<job>.out` log is saved as evidence. Blocker: Unity account and project storage path not set up yet. | Written, NOT tested on Unity | [____] |
| NLP-09 | One-page model write-ups | High | Plain-language explanation of how FinBERT and BERTopic work inside (Prof. Glatzer's Sprint 1 feedback). Done when `docs/models/finbert.md` and `docs/models/bertopic.md` exist, one page each. Blocker: none beyond time; reading the FinBERT and BERTopic papers. | Not started | [____] |
| NLP-10 | Hand-labeled test set | High | Labeling guidelines plus 300-500 AI mentions labeled on the 0-3 sophistication scale, split by call. About 100 double-labeled. Done when the `annotations` table holds at least 300 labels and Cohen's kappa is reported. Not a Sprint 3 finish if guidelines slip; first milestone is [____] labels. Blocker: who labels, and the labeling guidelines, are still open questions. | Not started | [____] |
| NLP-11 | First own-model baseline | Medium | TF-IDF + logistic regression buzzword-vs-substance classifier. Done when macro-F1 and a confusion matrix on the held-out split are saved, compared with the most-common-class and keyword-rule baselines. Depends on NLP-10. Blocker: NLP-10 labels. | Not started | [____] |
| NLP-08 | Model comparison and evaluation | Medium | Run the evaluation plan once NLP-10 and NLP-11 have results. Done when accuracy, macro-F1 and McNemar's test are reported for each model. Blocker: NLP-10 and NLP-11. | Plan written; not run | [____] |

## Potential work units

| ID | Name | Description |
| --- | --- | --- |
| POT-04 | LLM API access | Zero-shot sophistication baseline. Needs a purchase decision. |
| POT-05 | Seeking Alpha | Second transcript source. Needs a purchase decision. |
| POT-06 | Capital IQ premium | Mass export plug-in. Not planned. |
| POT-08 | Non-AI control group | Re-pull with `include_non_ai_calls: true` so every call has a 0/1 AI flag. Costs extra earningscalls.dev requests: [estimate ____]. Conflicts with current config; confirm first. |

## Notes

- Carry-over from Sprint 2: NLP-03 (Unity untested), NLP-08 (comparison not run), NLP-04 status [____].
- Data gaps found 2026-11-11: see "Known data gaps" in `docs/memory/project-state.md`.
- Purchases needed: [____].
- Timeline: [____].
