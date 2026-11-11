---
paths:
  - "src/ecnlp/sentiment.py"
  - "src/ecnlp/topics.py"
  - "src/ecnlp/nlp_scores.py"
  - "tests/test_nlp.py"
  - "environment.yml"
  - "notebooks/**/*"
  - "slurm/**/*"
---

# NLP model rules

## Models
- FinBERT (`ProsusAI/finbert`): sentiment, labels positive, neutral or negative with a score. Input is AI-mention context text.
- BERTopic: topics for AI mentions. Pipeline is sentence embeddings, then UMAP, then HDBSCAN, then c-TF-IDF keywords.
- Our own model: buzzword vs. substance on a 0-3 sophistication scale. Start with a baseline: TF-IDF + logistic regression, or Sentence-BERT embeddings + logistic regression. Fine-tuning DistilBERT is a later option that needs the Unity GPU.
- Run pretrained models on a small sample first, and document how each works before scaling up.

## Evaluation
- One held-out test set of 300-500 hand-labeled AI mentions. No model trains on it.
- About 100 mentions labeled by two annotators. Report Cohen's kappa.
- Split by call or company, never by sentence.
- Never use Financial PhraseBank as the test set (FinBERT was trained on it).
- Sentiment and sophistication: accuracy, macro-F1, confusion matrix, confidence intervals, McNemar's test between models.
- Topics: coherence (c_v or NPMI), topic diversity, and a human word-intrusion check.
- Sophistication baselines: the most common class, a keyword rule and a zero-shot model.
- Finance check: which model's scores better explain abnormal returns in `calls_master.csv`.

## Outputs
- Per-mention scores join on `sentence_id`. Call-level aggregates join to `calls_master.csv` on `call_id`.
- Planned tables: `tech_mentions`, `topics`, `annotations`, `call_scores`.
- Save model outputs to `data/processed/` and record the model name and version with each run.

## Unity cluster
- Use `slurm/` scripts. Set the project storage path before submitting. Never copy `.env` into shared storage.
