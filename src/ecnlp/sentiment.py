"""FinBERT sentiment for transcript sentences and AI-mention contexts.

Two unit types are scored and stored in the `sentiment` table:

* sentence    - every non-boilerplate sentence from `sentences` (operator turns
                are already dropped by `prepare`)
* ai_context  - each AI-term sentence with +/- N sentences from the same
                speaker turn (the SDS mention unit, see exports.build_ai_contexts)

Runs offline from SQLite and is incremental: a unit is re-scored only when
its text hash changes or the model changes, so an interrupted run on the
cluster resumes where it stopped. The model is only loaded if there is work.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import db
from .config import Config
from .exports import build_ai_contexts

log = logging.getLogger(__name__)

LABELS = ("positive", "negative", "neutral")
# texts -> (n, 3) array of probabilities in LABELS order
Scorer = Callable[[list[str]], np.ndarray]


def text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def pick_device(pref: str | None = "auto") -> str:
    if pref and pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def label_order(id2label: dict) -> list[int]:
    """Model output indices for LABELS.

    FinBERT variants disagree on label order and case: ProsusAI/finbert is
    positive/negative/neutral, yiyanghkust/finbert-tone is Neutral/Positive/Negative.
    """
    names = {int(i): str(lab).strip().lower() for i, lab in id2label.items()}
    missing = [lab for lab in LABELS if lab not in names.values()]
    if missing:
        raise ValueError(f"Model labels {names} do not include {missing}; is this a FinBERT sentiment model?")
    return [next(i for i, n in names.items() if n == lab) for lab in LABELS]


class FinBertScorer:
    """Batched FinBERT inference returning (n, 3) probabilities in LABELS order."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.order = label_order(self.model.config.id2label)
        log.info("Loaded %s on %s (labels %s)", model_name, self.device, self.model.config.id2label)

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")   # similar lengths -> less padding
        for start in range(0, len(texts), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                probs = self._torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            out[idx] = probs[:, self.order]
        return out


_warned_missing: set[tuple[str, ...]] = set()


def scoped_call_ids(cfg: Config, con: sqlite3.Connection) -> set[int]:
    """Calls the NLP stages cover: configured tickers, date_from..date_to, text pulled.

    The database can hold calls from earlier pilots or a wider pull; this keeps
    the topic model and the scores to the sample in config.yaml.
    """
    calls = db.read_df(con, "SELECT call_id, ticker, call_date_et FROM calls WHERE status='ok'")
    d = calls["call_date_et"].fillna("")
    keep = calls["ticker"].isin(cfg.tickers) & (d >= cfg.date_from) & (d <= cfg.date_to)
    missing = [t for t in cfg.tickers if t not in set(calls.loc[keep, "ticker"])]
    if missing and tuple(missing) not in _warned_missing:
        _warned_missing.add(tuple(missing))
        log.warning("No prepared calls for %s in %s..%s - run the `transcripts` and `prepare` stages first",
                    ",".join(missing), cfg.date_from, cfg.date_to)
    return set(calls.loc[keep, "call_id"].astype(int))


def load_units(cfg: Config, con: sqlite3.Connection) -> pd.DataFrame:
    """Units to score: unit_type, unit_id, call_id, text, text_hash."""
    scfg = cfg.nlp.get("sentiment") or {}
    min_words = int(scfg.get("min_words", 3))
    s = db.read_df(con, "SELECT sentence_id AS unit_id, call_id, text FROM sentences "
                        "WHERE is_boilerplate=0 AND n_words>=? ORDER BY call_id, segment_id, sent_idx", (min_words,))
    s.insert(0, "unit_type", "sentence")
    frames = [s]
    if scfg.get("score_ai_contexts", True):
        ctx = build_ai_contexts(con, window=int(scfg.get("context_window", 1)))
        if not ctx.empty:
            frames.append(pd.DataFrame({"unit_type": "ai_context", "unit_id": ctx["sentence_id"],
                                        "call_id": ctx["call_id"], "text": ctx["context_text"]}))
    units = pd.concat(frames, ignore_index=True)
    units = units[units["call_id"].isin(scoped_call_ids(cfg, con))].reset_index(drop=True)
    units["text_hash"] = units["text"].map(text_hash)
    return units


def score_sentiment(cfg: Config, con: sqlite3.Connection, scorer: Scorer | None = None) -> dict[str, Any]:
    scfg = cfg.nlp.get("sentiment") or {}
    model_name = scfg.get("model", "ProsusAI/finbert")
    chunk = int(scfg.get("chunk_size", 2000))

    units = load_units(cfg, con)
    done = db.read_df(con, "SELECT unit_type, unit_id, text_hash FROM sentiment WHERE model=?", (model_name,))
    key = ["unit_type", "unit_id", "text_hash"]
    todo = units.merge(done.assign(_done=1), on=key, how="left")
    todo = todo[todo["_done"].isna()].drop(columns="_done").reset_index(drop=True)
    stats = {"units": len(units), "already_scored": len(units) - len(todo), "scored": 0, "model": model_name}
    log.info("Sentiment: %d units, %d already scored with %s, %d to score",
             len(units), stats["already_scored"], model_name, len(todo))
    if todo.empty:
        return stats

    if scorer is None:
        scorer = FinBertScorer(model_name, device=cfg.nlp.get("device", "auto"),
                               batch_size=int(scfg.get("batch_size", 32)),
                               max_length=int(scfg.get("max_length", 256)))
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk]
        probs = np.asarray(scorer(part["text"].tolist()), dtype=float)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [{
            "unit_type": u.unit_type, "unit_id": u.unit_id, "model": model_name, "call_id": u.call_id,
            "text_hash": u.text_hash, "label": LABELS[int(np.argmax(p))],
            "p_positive": p[0], "p_negative": p[1], "p_neutral": p[2], "score": p[0] - p[1], "scored_at": now,
        } for u, p in zip(part.itertuples(index=False), probs)]
        db.upsert(con, "sentiment", rows)
        con.commit()   # commit per chunk so an interrupted run keeps its progress
        stats["scored"] += len(rows)
        log.info("Sentiment: %d / %d scored", stats["scored"], len(todo))
    return stats
