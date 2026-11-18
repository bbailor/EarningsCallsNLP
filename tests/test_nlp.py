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


def test_runinfo_versions_and_model_revision(tmp_path):
    from ecnlp.runinfo import model_revision, package_versions

    refs = tmp_path / "models--ProsusAI--finbert" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123def456\n")
    assert model_revision("ProsusAI/finbert", cache_dir=tmp_path) == "abc123def456"
    assert model_revision("not/cached", cache_dir=tmp_path) is None
    v = package_versions(("pandas", "no-such-package-xyz"))
    assert v["pandas"] == pd.__version__ and v["no-such-package-xyz"] is None


def test_nlp_run_records_environment_and_quiets_numba(cfg_path, monkeypatch, tmp_path):
    import json
    import logging

    import ecnlp.nlp_scores
    from ecnlp import pipeline
    from ecnlp.logging_utils import setup_logging

    setup_logging(tmp_path / "logs", "test")
    assert logging.getLogger("numba.core.ssa").getEffectiveLevel() >= logging.WARNING

    monkeypatch.setattr(ecnlp.nlp_scores, "build_call_scores", lambda cfg, con: {"calls": 0})
    cfg = load_config(str(cfg_path))
    assert pipeline.run(["scores"], cfg) == 0
    con = db.connect(cfg.paths.db)
    notes = json.loads(con.execute("SELECT notes FROM run_log WHERE stage='scores'").fetchone()[0])
    con.close()
    assert notes["result"] == {"calls": 0}
    assert notes["env"]["pkgs"]["pandas"] == pd.__version__
    assert notes["env"]["models"]["sentiment"].startswith("ProsusAI/finbert@")


def test_summary_stats_tables_and_figures(cfg_path, tmp_path):
    pytest.importorskip("matplotlib")
    from ecnlp.summary_stats import build_summary

    cfg = load_config(str(cfg_path))
    rng = np.random.default_rng(0)
    rows = []
    for i, (t, d) in enumerate([(t, f"2025-0{m}-15") for t in ("NVDA", "CAT") for m in range(1, 7)]):
        rows.append({"call_id": i, "ticker": t, "call_date_et": d, "n_words_ex_operator": 8000 + i,
                     "n_sentences": 400, "n_ai_sentences": 10 + i, "ai_sentences_per_1000_words": (10 + i) / 8,
                     "sent_all_mean": 0.2, "sent_prepared_mean": 0.3 + i / 100, "sent_qa_mean": 0.1,
                     "sent_ai_ctx_mean": 0.4, "sent_nonai_mean": 0.2, "sent_ai_minus_nonai": rng.normal(),
                     "topic_entropy": 0.7, "car_mkt_0_1": rng.normal() / 20})
    rows.append({"call_id": 99, "ticker": "MSFT", "call_date_et": "2025-03-01", "sent_all_mean": 0.1})  # not in config
    rows.append({"call_id": 98, "ticker": "NVDA", "call_date_et": "2025-08-01", "sent_all_mean": None})  # unscored
    master = tmp_path / "calls_master.csv"
    pd.DataFrame(rows).to_csv(master, index=False)

    res = build_summary(cfg, master_path=master)
    assert res["calls"] == 12 and res["tickers"] == 2 and res["files"] == 8
    out = cfg.paths.processed / "summary"
    overall = pd.read_csv(out / "summary_overall.csv").set_index("variable")
    assert overall.loc["n_ai_sentences", "n"] == 12
    assert overall.loc["n_ai_sentences", "mean"] == pytest.approx(15.5)
    by_t = pd.read_csv(out / "summary_by_ticker.csv").set_index("ticker")
    assert set(by_t.index) == {"NVDA", "CAT"} and by_t.loc["NVDA", "n_calls"] == 6
    corr = pd.read_csv(out / "correlations_with_car.csv")
    assert {"pearson", "spearman"} <= set(corr.columns) and (corr["n"] == 12).all()
    assert all((out / f).stat().st_size > 0 for f in ("fig1_ai_intensity_by_company.png",
                                                      "fig4_ai_tone_vs_return.png"))


def test_summary_requires_nlp_columns(cfg_path, tmp_path):
    from ecnlp.summary_stats import load_sample

    cfg = load_config(str(cfg_path))
    p = tmp_path / "m.csv"
    pd.DataFrame({"call_id": [1], "ticker": ["NVDA"], "call_date_et": ["2025-02-01"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="run `nlp` then `export`"):
        load_sample(cfg, p)
