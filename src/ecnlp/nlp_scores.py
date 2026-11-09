"""Call-level NLP variables from sentence sentiment and topic assignments.

`call_scores` is derived data (one row per call, rebuilt on every run like
event_returns) and joins to `calls` / `calls_master.csv` on call_id.

Sentiment columns, for each slice in SLICES (all, prepared, qa, exec,
analyst, nonai) plus ai_ctx (AI mentions with context):

    sent_<slice>_n         sentences scored
    sent_<slice>_mean      mean of p_positive - p_negative, in [-1, 1]
    sent_<slice>_net_tone  (n_pos - n_neg) / (n_pos + n_neg) using FinBERT labels
    sent_<slice>_pct_pos / _pct_neg

plus sent_ai_minus_nonai (AI-context tone relative to the rest of the call).

Topic columns: topic_entropy (normalized 0-1; 0 = one topic), topic_dominant,
topic_outlier_share and topic_<id>_share for the `wide_top_n` largest topics.
Every topic's share is in nlp_topic_shares.csv (long format).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from . import db
from .config import Config
from .exports import _write
from .sentiment import load_units, scoped_call_ids

log = logging.getLogger(__name__)

SLICES = {
    "all": None,
    "prepared": ("section", "prepared_remarks"),
    "qa": ("section", "qa"),
    "exec": ("speaker_type", "executive"),
    "analyst": ("speaker_type", "analyst"),
    "nonai": ("has_ai_term", 0),
}
META_COLS = ["call_id", "ticker", "call_key", "company_name", "fiscal_year", "fiscal_quarter",
             "call_date_et", "mentions_ai", "n_sentences", "n_words_ex_operator", "n_ai_sentences"]


def tone_by_call(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate unit-level FinBERT output (call_id, label, score) to one row per call."""
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="call_id"))
    d = df.assign(_pos=df["label"].eq("positive"), _neg=df["label"].eq("negative"))
    g = d.groupby("call_id")
    pos, neg = g["_pos"].sum(), g["_neg"].sum()
    out = pd.DataFrame({
        "n": g.size(),
        "mean": g["score"].mean(),
        "net_tone": (pos - neg) / (pos + neg).where(pos + neg > 0),
        "pct_pos": g["_pos"].mean(),
        "pct_neg": g["_neg"].mean(),
    })
    return out.add_prefix(prefix)


def _sentiment_frame(cfg: Config, con: sqlite3.Connection) -> pd.DataFrame:
    """Current units joined to their scores; stale scores (text changed) are ignored."""
    model = (cfg.nlp.get("sentiment") or {}).get("model", "ProsusAI/finbert")
    units = load_units(cfg, con)[["unit_type", "unit_id", "call_id", "text_hash"]]
    scores = db.read_df(con, "SELECT unit_type, unit_id, text_hash, label, score, p_positive, p_negative, p_neutral "
                             "FROM sentiment WHERE model=?", (model,))
    return units.merge(scores, on=["unit_type", "unit_id", "text_hash"], how="inner")


def sentiment_features(cfg: Config, con: sqlite3.Connection) -> pd.DataFrame:
    s = _sentiment_frame(cfg, con)
    meta = db.read_df(con, "SELECT sentence_id AS unit_id, section, speaker_type, has_ai_term FROM sentences")
    sent = s[s["unit_type"] == "sentence"].merge(meta, on="unit_id", how="left")
    parts = []
    for name, cond in SLICES.items():
        sub = sent if cond is None else sent[sent[cond[0]] == cond[1]]
        parts.append(tone_by_call(sub, f"sent_{name}_"))
    parts.append(tone_by_call(s[s["unit_type"] == "ai_context"], "sent_ai_ctx_"))
    out = pd.concat(parts, axis=1)
    if {"sent_ai_ctx_mean", "sent_nonai_mean"} <= set(out.columns):
        out["sent_ai_minus_nonai"] = out["sent_ai_ctx_mean"] - out["sent_nonai_mean"]
    return out


def topic_shares(cfg: Config, con: sqlite3.Connection) -> pd.DataFrame:
    """Long format: call_id, topic_id, topic_name, n, share."""
    corpus = (cfg.nlp.get("topics") or {}).get("corpus", "sentences")
    a = db.read_df(con, "SELECT call_id, topic_id FROM topic_assignments WHERE corpus=?", (corpus,))
    a = a[a["call_id"].isin(scoped_call_ids(cfg, con))]   # a failed refit can leave an older sample behind
    if a.empty:
        return pd.DataFrame(columns=["call_id", "topic_id", "topic_name", "n", "share"])
    n = a.groupby(["call_id", "topic_id"]).size().rename("n").reset_index()
    n["share"] = n["n"] / n.groupby("call_id")["n"].transform("sum")
    names = db.read_df(con, "SELECT topic_id, name AS topic_name FROM topics WHERE corpus=?", (corpus,))
    return n.merge(names, on="topic_id", how="left")[["call_id", "topic_id", "topic_name", "n", "share"]]


def topic_features(cfg: Config, shares: pd.DataFrame) -> pd.DataFrame:
    if shares.empty:
        return pd.DataFrame(index=pd.Index([], name="call_id"))
    wide_n = int((cfg.nlp.get("topics") or {}).get("wide_top_n", 20))
    real = shares[shares["topic_id"] >= 0].copy()
    real["share_ex_outlier"] = real["n"] / real.groupby("call_id")["n"].transform("sum")

    def entropy(p: pd.Series) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0

    out = pd.DataFrame({
        "topic_n_distinct": real.groupby("call_id").size(),
        "topic_entropy": real.groupby("call_id")["share_ex_outlier"].apply(entropy),
        "topic_dominant": real.loc[real.groupby("call_id")["n"].idxmax()].set_index("call_id")["topic_id"],
    })
    out["topic_outlier_share"] = shares[shares["topic_id"] == -1].set_index("call_id")["share"]
    out["topic_outlier_share"] = out["topic_outlier_share"].fillna(0.0)
    top = real.groupby("topic_id")["n"].sum().nlargest(wide_n).index
    wide = (real[real["topic_id"].isin(top)].pivot(index="call_id", columns="topic_id", values="share")
            .reindex(columns=sorted(top)).fillna(0.0))
    wide.columns = [f"topic_{t:02d}_share" for t in wide.columns]
    return out.join(wide, how="left")


def build_call_scores(cfg: Config, con: sqlite3.Connection) -> dict[str, Any]:
    """Rebuild `call_scores` and write the NLP exports to data/processed/."""
    sent = sentiment_features(cfg, con)
    shares = topic_shares(cfg, con)
    feats = sent.join(topic_features(cfg, shares), how="outer")
    feats.index.name = "call_id"
    feats = feats.reset_index()
    con.execute("DROP TABLE IF EXISTS call_scores")
    if not feats.empty:
        feats.to_sql("call_scores", con, index=False)
    con.commit()
    files = export_nlp(cfg, con, feats, shares)
    log.info("Call scores: %d calls x %d variables", len(feats), max(feats.shape[1] - 1, 0))
    return {"calls": len(feats), "variables": max(feats.shape[1] - 1, 0), **files}


def export_nlp(cfg: Config, con: sqlite3.Connection, feats: pd.DataFrame, shares: pd.DataFrame) -> dict[str, str]:
    out = cfg.paths.processed
    files: dict[str, str] = {}
    meta = db.read_df(con, f"SELECT {', '.join(META_COLS)} FROM calls WHERE status='ok'")
    meta = meta[meta["call_id"].isin(scoped_call_ids(cfg, con))]
    meta["ai_sentences_per_1000_words"] = (meta["n_ai_sentences"] /
                                           meta["n_words_ex_operator"].where(meta["n_words_ex_operator"] > 0) * 1000)
    panel = meta.merge(feats, on="call_id", how="left").sort_values(["ticker", "call_date_et"])
    files["call_scores"] = str(_write(panel, out / "nlp_call_scores.csv"))

    corpus = (cfg.nlp.get("topics") or {}).get("corpus", "sentences")
    topics = db.read_df(con, "SELECT * FROM topics WHERE corpus=? ORDER BY topic_id", (corpus,))
    files["topics"] = str(_write(topics, out / "nlp_topics.csv"))
    tickers = meta[["call_id", "ticker", "call_key", "call_date_et"]]
    files["topic_shares"] = str(_write(tickers.merge(shares, on="call_id"), out / "nlp_topic_shares.csv"))

    # Sentence-level detail for inspection and for re-aggregating differently.
    s = _sentiment_frame(cfg, con)
    s = s[s["unit_type"] == "sentence"].drop(columns=["unit_type", "text_hash"])
    sent = db.read_df(con, "SELECT sentence_id AS unit_id, section, speaker_type, has_ai_term, ai_terms, text "
                           "FROM sentences")
    detail = s.merge(sent, on="unit_id", how="left").merge(tickers, on="call_id", how="left")
    if corpus == "sentences":
        a = db.read_df(con, "SELECT unit_id, topic_id, prob AS topic_prob FROM topic_assignments WHERE corpus=?",
                       (corpus,))
        detail = detail.merge(a, on="unit_id", how="left")
    files["sentence_scores"] = str(_write(detail.rename(columns={"unit_id": "sentence_id"}),
                                          out / "nlp_sentence_scores.parquet"))
    for k, v in files.items():
        log.info("export %-15s -> %s", k, v)
    return files
