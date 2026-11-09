"""BERTopic topic model over the transcript corpus.

One global model is fit across every call in the sample, so topic ids are
comparable across companies and quarters. Results go to `topics` (one row
per topic) and `topic_assignments` (one row per document), both keyed by
`corpus`:

* sentences    - non-boilerplate sentences with at least `min_words` words
* ai_contexts  - AI-term sentences with surrounding context (AI themes only)

Unlike sentiment this stage refits from scratch on every run: adding calls
changes the clusters. Sentence embeddings are cached under
<data_dir>/models/, so re-tuning min_topic_size / nr_topics skips the
embedding step. The fitted model is saved there too.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import db
from .config import Config
from .exports import build_ai_contexts
from .sentiment import pick_device, scoped_call_ids

log = logging.getLogger(__name__)

CORPORA = ("sentences", "ai_contexts")
# docs -> (topic id per doc, probability per doc or None, topic info with
# columns topic_id, name, n_docs, top_words, representative_doc)
TopicFitter = Callable[[list[str]], tuple[list[int], Any, pd.DataFrame]]

# Earnings-call filler that would otherwise dominate the topic words.
DEFAULT_STOP_WORDS = [
    "quarter", "quarters", "year", "years", "think", "going", "really", "know", "just", "like", "thank",
    "thanks", "question", "questions", "yes", "yeah", "okay", "lot", "bit", "said", "say", "kind", "sort",
    "great", "good", "want", "look", "looking", "way", "things", "thing", "today", "right", "time", "mean",
    "maybe", "obviously", "certainly", "actually", "continue", "continued", "terms", "also", "would", "could",
    # contraction fragments left by the tokenizer ("don't" -> "don", "we'll" -> "ll")
    "don", "ll", "ve", "re", "doesn", "didn", "isn", "aren", "wasn", "won", "let",
]


def load_corpus(cfg: Config, con: sqlite3.Connection, corpus: str) -> pd.DataFrame:
    """Documents to model: unit_id, call_id, text (scoped to the configured sample)."""
    docs = _load_corpus(cfg, con, corpus)
    return docs[docs["call_id"].isin(scoped_call_ids(cfg, con))].reset_index(drop=True)


def _load_corpus(cfg: Config, con: sqlite3.Connection, corpus: str) -> pd.DataFrame:
    tcfg = cfg.nlp.get("topics") or {}
    if corpus == "sentences":
        return db.read_df(con, "SELECT sentence_id AS unit_id, call_id, text FROM sentences "
                               "WHERE is_boilerplate=0 AND n_words>=? ORDER BY call_id, segment_id, sent_idx",
                          (int(tcfg.get("min_words", 8)),))
    if corpus == "ai_contexts":
        ctx = build_ai_contexts(con, window=int((cfg.nlp.get("sentiment") or {}).get("context_window", 1)))
        if ctx.empty:
            return pd.DataFrame(columns=["unit_id", "call_id", "text"])
        return pd.DataFrame({"unit_id": ctx["sentence_id"], "call_id": ctx["call_id"], "text": ctx["context_text"]})
    raise ValueError(f"nlp.topics.corpus must be one of {CORPORA}, got {corpus!r}")


def bertopic_fitter(cfg: Config, corpus: str) -> TopicFitter:
    tcfg = cfg.nlp.get("topics") or {}
    emb_name = tcfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    seed = int(tcfg.get("seed", 42))
    model_dir = cfg.paths.models

    def fit(docs: list[str]):
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
        from umap import UMAP

        device = pick_device(cfg.nlp.get("device", "auto"))
        embedder = SentenceTransformer(emb_name, device=device)
        embeddings = _cached_embeddings(model_dir, emb_name, docs, embedder)

        stop = sorted(set(ENGLISH_STOP_WORDS) | set(DEFAULT_STOP_WORDS) | set(tcfg.get("extra_stop_words") or []))
        vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=int(tcfg.get("min_df", 1)))
        nr_topics = tcfg.get("nr_topics") or None
        model = BERTopic(
            embedding_model=embedder,
            umap_model=UMAP(n_neighbors=int(tcfg.get("umap_n_neighbors", 15)), n_components=5, min_dist=0.0,
                            metric="cosine", random_state=seed),
            hdbscan_model=HDBSCAN(min_cluster_size=int(tcfg.get("min_topic_size", 30)), metric="euclidean",
                                  cluster_selection_method="eom", prediction_data=True),
            vectorizer_model=vectorizer,
            top_n_words=int(tcfg.get("top_n_words", 10)),
            nr_topics=int(nr_topics) if str(nr_topics).isdigit() else nr_topics,
        )
        topics, probs = model.fit_transform(docs, embeddings)
        if tcfg.get("reduce_outliers", True) and -1 in topics and len(set(topics)) > 1:
            # Reassign outliers (-1) to their nearest topic; probabilities no longer apply to those docs.
            new = model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
            probs = np.where(np.asarray(new) != np.asarray(topics), np.nan, np.asarray(probs, dtype=float))
            topics = new
            model.update_topics(docs, topics=topics, vectorizer_model=vectorizer,
                                top_n_words=int(tcfg.get("top_n_words", 10)))

        info = model.get_topic_info()
        rows = []
        for r in info.itertuples(index=False):
            words = model.get_topic(r.Topic) or []
            rep = getattr(r, "Representative_Docs", None)
            rows.append({"topic_id": int(r.Topic), "name": r.Name, "n_docs": int(r.Count),
                         "top_words": "|".join(w for w, _ in words),
                         "representative_doc": rep[0] if isinstance(rep, list) and rep else None})
        try:
            save_to = model_dir / f"bertopic_{corpus}"
            model.save(str(save_to), serialization="safetensors", save_ctfidf=True, save_embedding_model=emb_name)
            log.info("BERTopic model saved to %s", save_to)
        except Exception as exc:  # saving is a convenience, not required for the scores
            log.warning("Could not save BERTopic model: %s", exc)
        return list(topics), probs, pd.DataFrame(rows)

    return fit


def _cached_embeddings(model_dir: Path, emb_name: str, docs: list[str], embedder) -> np.ndarray:
    h = hashlib.sha1(emb_name.encode())
    for d in docs:
        h.update(d.encode("utf-8"))
        h.update(b"\0")
    path = model_dir / f"embeddings_{h.hexdigest()[:16]}.npy"
    if path.exists():
        log.info("Using cached embeddings %s", path.name)
        return np.load(path)
    log.info("Embedding %d documents with %s", len(docs), emb_name)
    emb = embedder.encode(docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def fit_topics(cfg: Config, con: sqlite3.Connection, fitter: TopicFitter | None = None) -> dict[str, Any]:
    tcfg = cfg.nlp.get("topics") or {}
    corpus = tcfg.get("corpus", "sentences")
    docs = load_corpus(cfg, con, corpus)
    min_docs = int(tcfg.get("min_docs", 200))
    if len(docs) < min_docs:
        log.warning("Topics: only %d %s documents (< nlp.topics.min_docs=%d); skipping BERTopic",
                    len(docs), corpus, min_docs)
        return {"corpus": corpus, "docs": len(docs), "topics": 0, "skipped": True}

    log.info("Topics: fitting BERTopic on %d %s documents", len(docs), corpus)
    fitter = fitter or bertopic_fitter(cfg, corpus)
    topics, probs, info = fitter(docs["text"].tolist())
    if len(topics) != len(docs):
        raise ValueError(f"Topic fitter returned {len(topics)} assignments for {len(docs)} documents")
    probs = np.full(len(docs), np.nan) if probs is None else np.asarray(probs, dtype=float)
    if probs.ndim == 2:   # full topic distribution -> probability of the assigned topic
        probs = probs.max(axis=1)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_label = "bertopic:" + tcfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    db.replace_where(con, "topics", "corpus=?", (corpus,),
                     [{**r, "corpus": corpus, "model": model_label, "fitted_at": now}
                      for r in info.to_dict("records")])
    db.replace_where(con, "topic_assignments", "corpus=?", (corpus,), [
        {"corpus": corpus, "unit_id": u, "call_id": c, "topic_id": int(t), "prob": p}
        for u, c, t, p in zip(docs["unit_id"], docs["call_id"], topics, probs)])
    con.commit()
    n_topics = int((info["topic_id"] >= 0).sum()) if len(info) else 0
    outliers = float(np.mean(np.asarray(topics) == -1))
    log.info("Topics: %d topics, %.1f%% outliers", n_topics, 100 * outliers)
    return {"corpus": corpus, "docs": len(docs), "topics": n_topics, "outlier_share": round(outliers, 4)}
