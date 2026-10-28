"""Clean pulled segments and split them into sentences ready for FinBERT/BERTopic.

Runs entirely offline from the SQLite tables, so it can be re-run (e.g. after
changing the keyword list) without touching the API.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

from . import db
from .cleaning import build_ai_matcher, is_boilerplate, normalize_text, split_sentences, word_count
from .config import Config

log = logging.getLogger(__name__)


def prepare_text(cfg: Config, con: sqlite3.Connection, only_new: bool = False) -> dict[str, Any]:
    match_ai = build_ai_matcher(cfg.cleaning.get("ai_terms_case_sensitive") or [],
                                cfg.cleaning.get("ai_terms_case_insensitive") or [])
    drop_operator = bool(cfg.cleaning.get("drop_operator_segments", True))

    q = "SELECT call_id FROM calls WHERE status='ok'"
    if only_new:
        q += " AND (n_sentences IS NULL)"
    call_ids = [r[0] for r in con.execute(q)]
    totals = {"calls": 0, "sentences": 0, "ai_sentences": 0}

    for cid in call_ids:
        segs = db.read_df(con, "SELECT * FROM segments WHERE call_id=? ORDER BY component_order", (cid,))
        seg_updates, sent_rows = [], []
        for s in segs.itertuples(index=False):
            clean = normalize_text(s.text_raw)
            seg_updates.append({"segment_id": s.segment_id, "text_clean": clean, "n_words": word_count(clean)})
            if drop_operator and s.is_operator:
                continue
            for i, sent in enumerate(split_sentences(clean)):
                terms = match_ai(sent)
                sent_rows.append({
                    "sentence_id": f"{s.segment_id}-{i}",
                    "call_id": cid,
                    "segment_id": s.segment_id,
                    "sent_idx": i,
                    "section": s.section,
                    "speaker_type": s.speaker_type,
                    "text": sent,
                    "n_words": word_count(sent),
                    "is_boilerplate": int(is_boilerplate(sent)),
                    "has_ai_term": int(bool(terms)),
                    "ai_terms": "|".join(terms) or None,
                })
        for u in seg_updates:
            con.execute("UPDATE segments SET text_clean=?, n_words=? WHERE segment_id=?",
                        (u["text_clean"], u["n_words"], u["segment_id"]))
        db.replace_where(con, "sentences", "call_id=?", (cid,), sent_rows)
        n_ai = sum(r["has_ai_term"] for r in sent_rows if not r["is_boilerplate"])
        con.execute("UPDATE calls SET n_sentences=?, n_ai_sentences=? WHERE call_id=?", (len(sent_rows), n_ai, cid))
        totals["calls"] += 1
        totals["sentences"] += len(sent_rows)
        totals["ai_sentences"] += n_ai
        if totals["calls"] % 100 == 0:
            con.commit()
    con.commit()

    # Calls that matched the API's AI search but none of our keywords deserve a look
    # (keyword list gaps or a stemming quirk on the server side).
    gaps = con.execute("SELECT COUNT(*) FROM calls WHERE status='ok' AND mentions_ai=1 AND n_ai_sentences=0").fetchone()[0]
    if gaps:
        log.warning("%d AI-matched calls contain none of the configured AI keywords - review the dictionary (POT-02)", gaps)
    log.info("Text prepared: %s", totals)
    return totals
