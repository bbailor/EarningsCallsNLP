"""Sampling, persistence, and agreement statistics for human annotations."""
from __future__ import annotations

import random
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import db


LABELS = {
    0: "0 — No substantive technology claim",
    1: "1 — Generic or aspirational mention",
    2: "2 — Concrete use, capability, or investment",
    3: "3 — Material or strategic technology impact",
}


def create_annotation_sample(
    con: sqlite3.Connection,
    sample_size: int = 100,
    double_fraction: float = 0.2,
    seed: int = 491,
) -> dict[str, Any]:
    """Create a reproducible sample from prepared AI-mention sentences."""
    if sample_size < 1:
        raise ValueError("sample_size must be greater than zero")
    if not 0 <= double_fraction <= 1:
        raise ValueError("double_fraction must be between zero and one")

    sentences = db.read_df(
        con,
        """
        SELECT sentence_id, call_id, segment_id, sent_idx, section, speaker_type, text,
               ai_terms, is_boilerplate, has_ai_term
        FROM sentences
        WHERE has_ai_term = 1 AND is_boilerplate = 0
          AND COALESCE(speaker_type, '') != 'operator'
        ORDER BY sentence_id
        """,
    )
    if sentences.empty:
        raise ValueError("No eligible AI-mention sentences. Run the transcripts and prepare stages first.")

    neighbors = db.read_df(
        con,
        """
        SELECT sentence_id, segment_id, sent_idx, text
        FROM sentences
        WHERE is_boilerplate = 0 AND COALESCE(speaker_type, '') != 'operator'
        ORDER BY segment_id, sent_idx
        """,
    )
    context_by_id: dict[str, str] = {}
    for _, group in neighbors.groupby("segment_id", sort=False):
        rows = list(group.itertuples(index=False))
        for index, row in enumerate(rows):
            low, high = max(0, index - 1), min(len(rows), index + 2)
            context_by_id[row.sentence_id] = " ".join(item.text for item in rows[low:high])

    candidates = sentences.to_dict("records")
    rng = random.Random(seed)
    selected = rng.sample(candidates, min(sample_size, len(candidates)))
    double_count = round(len(selected) * double_fraction)
    double_ids = {
        row["sentence_id"]
        for row in rng.sample(selected, double_count)
    } if double_count else set()

    batch_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute(
        "INSERT INTO annotation_samples "
        "(batch_id, created_at, requested_size, actual_size, double_fraction, seed) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (batch_id, created_at, sample_size, len(selected), double_fraction, seed),
    )
    con.executemany(
        "INSERT INTO annotation_sample_items "
        "(batch_id, sentence_id, sample_index, double_label, context_text) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (
                batch_id,
                row["sentence_id"],
                index,
                int(row["sentence_id"] in double_ids),
                context_by_id.get(row["sentence_id"], row["text"]),
            )
            for index, row in enumerate(selected, start=1)
        ],
    )
    con.commit()
    return {
        "batch_id": batch_id,
        "requested": sample_size,
        "actual": len(selected),
        "double_labeled": len(double_ids),
        "available": len(candidates),
    }


def sample_items(con: sqlite3.Connection, batch_id: str) -> pd.DataFrame:
    """Return sample passages with transcript metadata and current label counts."""
    return db.read_df(
        con,
        """
        SELECT i.sample_index, i.sentence_id, i.double_label,
               s.call_id, s.section, s.speaker_type, s.text AS anchor_sentence, s.ai_terms,
               c.ticker, c.call_key, c.call_date_et,
               i.context_text,
               COUNT(a.annotator) AS annotation_count
        FROM annotation_sample_items i
        JOIN sentences s ON s.sentence_id = i.sentence_id
        JOIN calls c ON c.call_id = s.call_id
        LEFT JOIN annotations a
          ON a.batch_id = i.batch_id AND a.sentence_id = i.sentence_id
        WHERE i.batch_id = ?
        GROUP BY i.sample_index, i.sentence_id, i.double_label, s.call_id, s.section,
                 s.speaker_type, s.text, s.ai_terms, c.ticker, c.call_key, c.call_date_et,
                 i.context_text
        ORDER BY i.sample_index
        """,
        (batch_id,),
    )


def save_annotation(
    con: sqlite3.Connection,
    batch_id: str,
    sentence_id: str,
    annotator: str,
    label: int,
    notes: str | None = None,
) -> None:
    annotator = annotator.strip()
    if not annotator:
        raise ValueError("Annotator name or initials are required.")
    if label not in LABELS:
        raise ValueError("Label must be an integer from 0 to 3.")
    exists = con.execute(
        "SELECT 1 FROM annotation_sample_items WHERE batch_id=? AND sentence_id=?",
        (batch_id, sentence_id),
    ).fetchone()
    if not exists:
        raise ValueError("Passage is not part of this annotation sample.")
    con.execute(
        """
        INSERT INTO annotations (batch_id, sentence_id, annotator, label, notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(batch_id, sentence_id, annotator) DO UPDATE SET
            label=excluded.label, notes=excluded.notes, updated_at=excluded.updated_at
        """,
        (
            batch_id,
            sentence_id,
            annotator,
            label,
            notes.strip() if notes and notes.strip() else None,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    con.commit()


def annotators_for_batch(con: sqlite3.Connection, batch_id: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT annotator FROM annotations WHERE batch_id=? ORDER BY annotator",
            (batch_id,),
        )
    ]


def cohen_kappa(
    con: sqlite3.Connection,
    batch_id: str,
    annotator_a: str,
    annotator_b: str,
) -> dict[str, Any]:
    """Calculate unweighted Cohen's kappa for jointly labeled double-label items."""
    if annotator_a == annotator_b:
        raise ValueError("Choose two different annotators.")
    pairs = db.read_df(
        con,
        """
        SELECT a.sentence_id, a.label AS label_a, b.label AS label_b
        FROM annotation_sample_items i
        JOIN annotations a
          ON a.batch_id = i.batch_id AND a.sentence_id = i.sentence_id
         AND a.annotator = ?
        JOIN annotations b
          ON b.batch_id = i.batch_id AND b.sentence_id = i.sentence_id
         AND b.annotator = ?
        WHERE i.batch_id = ? AND i.double_label = 1
        """,
        (annotator_a, annotator_b, batch_id),
    )
    n = len(pairs)
    if not n:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}

    observed = float((pairs["label_a"] == pairs["label_b"]).mean())
    expected = sum(
        float((pairs["label_a"] == label).mean()) * float((pairs["label_b"] == label).mean())
        for label in LABELS
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }
