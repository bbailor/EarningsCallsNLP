from __future__ import annotations

import pytest

from ecnlp import annotation, db


@pytest.fixture
def annotated_db(tmp_path):
    con = db.connect(tmp_path / "annotations.sqlite")
    con.execute("INSERT INTO companies (ticker) VALUES ('TEST')")
    con.execute(
        "INSERT INTO calls (call_id, ticker, call_key, call_date_et) VALUES (1, 'TEST', 'TEST_2025Q1', '2025-01-01')"
    )
    con.execute(
        "INSERT INTO segments (segment_id, call_id, component_order, speaker_type) "
        "VALUES ('seg', 1, 0, 'executive')"
    )
    rows = []
    for index in range(10):
        rows.append(
            (
                f"sent-{index}",
                1,
                "seg",
                index,
                "prepared_remarks",
                "executive",
                f"Sentence {index} discusses AI and business context.",
                8,
                0,
                int(index % 2 == 0),
                "AI" if index % 2 == 0 else None,
            )
        )
    con.executemany(
        "INSERT INTO sentences "
        "(sentence_id, call_id, segment_id, sent_idx, section, speaker_type, text, n_words, "
        "is_boilerplate, has_ai_term, ai_terms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    yield con
    con.close()


def test_sample_has_requested_size_context_and_double_label_fraction(annotated_db):
    result = annotation.create_annotation_sample(annotated_db, sample_size=5, double_fraction=0.2, seed=7)
    sample = annotation.sample_items(annotated_db, result["batch_id"])

    assert result["actual"] == 5
    assert result["double_labeled"] == 1
    assert len(sample) == 5
    assert sample["sentence_id"].is_unique
    assert sample["double_label"].sum() == 1
    assert sample["context_text"].str.len().ge(sample["anchor_sentence"].str.len()).all()


def test_sample_reports_missing_prepared_sentences(annotated_db):
    annotated_db.execute("DELETE FROM sentences")
    with pytest.raises(ValueError, match="No eligible AI-mention sentences"):
        annotation.create_annotation_sample(annotated_db)


def test_annotations_are_independent_and_kappa_uses_double_labeled_items(annotated_db):
    result = annotation.create_annotation_sample(annotated_db, sample_size=10, double_fraction=0.4, seed=4)
    sample = annotation.sample_items(annotated_db, result["batch_id"])
    double_items = sample.loc[sample["double_label"] == 1, "sentence_id"].tolist()
    assert len(double_items) == 2

    annotation.save_annotation(annotated_db, result["batch_id"], double_items[0], "A", 0)
    annotation.save_annotation(annotated_db, result["batch_id"], double_items[1], "A", 1)
    annotation.save_annotation(annotated_db, result["batch_id"], double_items[0], "B", 0)
    annotation.save_annotation(annotated_db, result["batch_id"], double_items[1], "B", 2)
    annotation.save_annotation(annotated_db, result["batch_id"], double_items[1], "A", 1, "Updated")

    stats = annotation.cohen_kappa(annotated_db, result["batch_id"], "A", "B")
    assert stats["n"] == 2
    assert stats["observed_agreement"] == 0.5
    assert stats["expected_agreement"] == 0.25
    assert stats["kappa"] == pytest.approx(1 / 3)
    assert annotated_db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 4
    assert annotation.annotators_for_batch(annotated_db, result["batch_id"]) == ["A", "B"]
