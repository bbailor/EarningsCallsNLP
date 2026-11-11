# Earnings Call NLP (Team GPT Gains)

Capstone project: measure whether AI talk on earnings calls is buzzword or substance, and test whether the market reacts differently. Advisor: Prof. Glatzer. Repo branch: `master`.

Current status, decisions and sprint history are in these files. Read them before planning work:
@docs/memory/project-state.md
@docs/memory/decisions.md
@docs/memory/sprint-log.md

## Commands (run from repo root, venv active)
- Set up: `python -m venv .venv`, then `.venv\Scripts\activate` (Windows) and `pip install -e . -r requirements.txt` (add `pip install -e ".[nlp]"` for FinBERT/BERTopic)
- Check the live API (about 4 requests): `python -m ecnlp.pipeline check`
- Small run: `python -m ecnlp.pipeline all --tickers NVDA,CAT`
- Full run: `python -m ecnlp.pipeline all` (transcripts -> market -> prepare -> events -> export)
- NLP stages: `python -m ecnlp.pipeline nlp` (sentiment -> topics -> scores); single stages: `sentiment`, `topics`, `scores`
- Tests: `pytest` (16 tests at end of Sprint 1, 22 after the NLP stages; all must pass before a commit)

## Layout
- `src/ecnlp/` is the package: earningscalls.dev client, yfinance pull, cleaning, event returns, exports, pipeline CLI. NLP code is `sentiment.py` (FinBERT), `topics.py` (BERTopic) and `nlp_scores.py`.
- `tests/` has `test_pipeline.py` and `test_nlp.py` (mocked APIs only)
- `config.yaml` holds tickers, `date_from`, `provider`, `naive_timestamp_tz`, `include_non_ai_calls` and the AI keyword list
- `.env` holds the API key. Never commit it. `.env.example` is the template.
- `scripts/` has read-only helpers: `demo.py` (step-by-step NLP demo) and `view_data.py` (coverage, calls, run log). Run from the repo root.
- `data/earnings_nlp.sqlite` is the source of truth. Exports in `data/processed/` are rebuilt from it.
- `data/raw/` is the API cache. Deleting it costs quota.
- `data/logs/` has one log per run
- `slurm/` has the Unity sbatch scripts (`pull_data.sbatch`, `nlp.sbatch`); `environment.yml` is at the repo root (written, not yet tested on Unity)
- `docs/memory/` is the shared project memory; `docs/claude-project-instructions.md` is the text pasted into the claude.ai project; `docs/sprints/` holds work plans and status reports

## Key data files
- `data/processed/calls_master.csv`: one row per call, with returns. NLP scores join here on `call_id`.
- `data/processed/nlp_ai_contexts.parquet`: one row per AI mention with one sentence of context on each side. Main NLP input.
- `data/processed/nlp_sentences.parquet`: all non-boilerplate sentences
- `data/processed/coverage_report.csv`: per-ticker attrition. Check it first after any run.

## Working rules
- Never invent results. Unknown evidence is written as `[____]`.
- Keep `sentence_id` and `call_id` stable. Hand labels and NLP scores attach to them.
- New DB tables go through `src/ecnlp/db.py`. Dashboard code reads through a query module, not raw SQL.
- After finishing a work unit, update `docs/memory/project-state.md` and add a line to `docs/memory/sprint-log.md`.
- After any decision about scope, models or data, add a dated entry to `docs/memory/decisions.md`.
