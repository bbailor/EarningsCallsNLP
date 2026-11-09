"""NLP stages with fake FinBERT / BERTopic so tests stay offline and fast."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import responses

from conftest import install_fake_api
from ecnlp import db
from ecnlp.config import load_config
from ecnlp.earningscalls_client import EarningsCallsClient
from ecnlp.nlp_scores import build_call_scores, tone_by_call
from ecnlp.prepare import prepare_text
from ecnlp.sentiment import label_order, score_sentiment
from ecnlp.topics import fit_topics
from ecnlp.transcripts import pull_transcripts


class FakeScorer:
    """Positive for growth words, negative for declines, else neutral."""

    def __init__(self):
        self.seen = 0

    def __call__(self, texts):
        self.seen += len(texts)
        out = []
        for t in texts:
            t = t.lower()
            if any(w in t for w in ("grew", "reduced inventory", "drives")):
                out.append([0.8, 0.1, 0.1])
            elif any(w in t for w in ("differ", "decline")):
                out.append([0.1, 0.7, 0.2])
            else:
                out.append([0.1, 0.1, 0.8])
        return np.array(out)


def fake_fitter(docs):
    topics = [0 if "ai" in d.lower() or "learning" in d.lower() else (1 if i % 3 else -1) for i, d in enumerate(docs)]
    info = pd.DataFrame({"topic_id": [-1, 0, 1], "name": ["-1_outlier", "0_ai_learning", "1_revenue"],
                         "n_docs": [topics.count(-1), topics.count(0), topics.count(1)],
                         "top_words": ["the", "ai|learning", "revenue|growth"], "representative_doc": [None, None, None]})
    return topics, np.full(len(docs), 0.9), info


@pytest.fixture
def prepared(cfg_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = load_config(cfg_path)
    cfg.nlp["topics"]["min_docs"] = 1
    cfg.nlp["topics"]["min_words"] = 3
    cfg.paths.ensure()
    con = db.connect(cfg.paths.db)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        install_fake_api(rsps, {})
        pull_transcripts(cfg, EarningsCallsClient("test-key", cache_dir=cfg.paths.raw_ec), con, remaining_quota=10000)
    prepare_text(cfg, con)
    yield cfg, con
    con.close()


def test_label_order_handles_finbert_variants():
    assert label_order({0: "positive", 1: "negative", 2: "neutral"}) == [0, 1, 2]       # ProsusAI/finbert
    assert label_order({"0": "Neutral", "1": "Positive", "2": "Negative"}) == [1, 2, 0]  # finbert-tone
    with pytest.raises(ValueError):
        label_order({0: "LABEL_0", 1: "LABEL_1"})


def test_tone_by_call():
    df = pd.DataFrame({"call_id": [1, 1, 1, 2], "label": ["positive", "negative", "neutral", "neutral"],
                       "score": [0.8, -0.6, 0.0, 0.0]})
    out = tone_by_call(df, "x_").loc
    assert out[1, "x_n"] == 3 and out[1, "x_net_tone"] == 0 and out[1, "x_mean"] == pytest.approx(0.2 / 3)
    assert np.isnan(out[2, "x_net_tone"]) and out[2, "x_pct_pos"] == 0


def test_sentiment_is_incremental(prepared):
    cfg, con = prepared
    scorer = FakeScorer()
    first = score_sentiment(cfg, con, scorer)
    assert first["scored"] > 0 and scorer.seen == first["scored"]
    types = {r[0] for r in con.execute("SELECT DISTINCT unit_type FROM sentiment")}
    assert types == {"sentence", "ai_context"}
    assert not con.execute("SELECT 1 FROM sentiment s JOIN sentences x ON x.sentence_id = s.unit_id "
                           "WHERE s.unit_type='sentence' AND x.is_boilerplate=1").fetchone()

    again = score_sentiment(cfg, con, FakeScorer())          # nothing changed -> nothing re-scored
    assert again["scored"] == 0 and again["already_scored"] == first["units"]

    con.execute("UPDATE sentences SET text = text || ' Margins grew.' WHERE sentence_id = "
                "(SELECT sentence_id FROM sentences WHERE is_boilerplate=0 AND has_ai_term=0 LIMIT 1)")
    # the edited sentence, plus the AI context it sits in (if any), are re-scored
    assert score_sentiment(cfg, con, FakeScorer())["scored"] in (1, 2)


def test_scope_excludes_other_tickers(prepared):
    cfg, con = prepared
    cfg.tickers = ["NVDA"]
    score_sentiment(cfg, con, FakeScorer())
    tickers = {r[0] for r in con.execute("SELECT DISTINCT c.ticker FROM sentiment s JOIN calls c USING(call_id)")}
    assert tickers == {"NVDA"}


def test_topics_and_call_scores(prepared):
    cfg, con = prepared
    score_sentiment(cfg, con, FakeScorer())
    res = fit_topics(cfg, con, fake_fitter)
    assert res["topics"] == 2 and res["docs"] > 0
    out = build_call_scores(cfg, con)

    panel = pd.read_csv(out["call_scores"]).set_index("call_id")
    assert set(panel.index) == {1001, 1002, 2002, 3001}
    nv = panel.loc[1001]
    assert nv["ticker"] == "NVDA" and nv["sent_all_n"] > 0
    assert nv["sent_exec_mean"] > 0                      # "Revenue grew" / "drives 30%" are executive
    assert nv["sent_analyst_n"] == 1 and nv["sent_ai_ctx_n"] >= 1
    assert {"sent_qa_net_tone", "sent_prepared_mean", "sent_ai_minus_nonai", "topic_entropy",
            "topic_00_share", "topic_01_share", "topic_outlier_share"} <= set(panel.columns)
    assert 0 <= nv["topic_entropy"] <= 1

    shares = pd.read_csv(out["topic_shares"])
    assert np.allclose(shares.groupby("call_id")["share"].sum(), 1.0)
    detail = pd.read_parquet(out["sentence_scores"])
    assert {"sentence_id", "label", "score", "topic_id", "ticker"} <= set(detail.columns)
    assert detail["sentence_id"].is_unique

    # call_scores flows into calls_master for the market cross-reference later
    from ecnlp.exports import export_all
    master = pd.read_csv(export_all(cfg, con)["calls_master"])
    assert "sent_all_mean" in master.columns and master["call_id"].is_unique


def test_topics_skip_small_corpus(prepared):
    cfg, con = prepared
    cfg.nlp["topics"]["min_docs"] = 10_000
    assert fit_topics(cfg, con, fake_fitter)["skipped"]
