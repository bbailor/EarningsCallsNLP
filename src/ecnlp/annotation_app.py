"""Streamlit interface for the human passage-annotation prototype."""
from __future__ import annotations

import streamlit as st

from ecnlp import annotation, db
from ecnlp.config import ConfigError, load_config


def main() -> None:
    st.set_page_config(page_title="Earnings Call Passage Annotation", layout="wide")
    st.title("Earnings Call Technology Annotation")
    st.caption("Label the AI-mention passage using the level 0–3 codebook.")
    with st.expander("Label definitions"):
        for level, definition in annotation.LABELS.items():
            st.markdown(f"- **{definition}**")
        st.caption(
            "Choose the strongest level supported by the displayed passage. "
            "Do not infer company use or impact from outside knowledge."
        )

    config_path = st.sidebar.text_input("Config file", value="config.yaml")
    annotator = st.sidebar.text_input("Annotator name or initials").strip()
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        st.error(str(exc))
        return

    cfg.paths.ensure()
    con = db.connect(cfg.paths.db)
    try:
        with st.expander("Create a labeling sample", expanded=False):
            st.write(
                "Samples are drawn from prepared AI-mention sentences. "
                "The default is 100 passages with 20% selected for double-labeling."
            )
            sample_size = st.number_input("Passage target", min_value=1, value=100, step=10)
            double_percent = st.number_input(
                "Double-label percentage", min_value=0, max_value=100, value=20, step=5
            )
            seed = st.number_input("Random seed", min_value=0, value=491, step=1)
            if st.button("Generate sample"):
                try:
                    result = annotation.create_annotation_sample(
                        con, int(sample_size), float(double_percent) / 100, int(seed)
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(
                        f"Created {result['actual']} passages "
                        f"({result['double_labeled']} marked for double-labeling)."
                    )
                    st.rerun()

        batches = con.execute(
            "SELECT batch_id, created_at, actual_size FROM annotation_samples ORDER BY created_at DESC"
        ).fetchall()
        if not batches:
            st.info("Generate an annotation sample after transcripts have been pulled and prepared.")
            return

        batch_by_label = {
            f"{created_at} — {size} passages — {batch_id[:8]}": batch_id
            for batch_id, created_at, size in batches
        }
        selected_batch_label = st.selectbox("Sample batch", list(batch_by_label))
        batch_id = batch_by_label[selected_batch_label]
        items = annotation.sample_items(con, batch_id)
        if items.empty:
            st.warning("This sample contains no passages.")
            return

        saved_annotations = int(items["annotation_count"].sum())
        expected_annotations = len(items) + int(items["double_label"].sum())
        st.progress(
            min(float(saved_annotations) / max(float(expected_annotations), 1.0), 1.0),
            text=f"{saved_annotations} of {expected_annotations} expected labels saved",
        )
        options = items["sentence_id"].tolist()
        selected_sentence = st.selectbox(
            "Passage",
            options,
            format_func=lambda sentence_id: _passage_option(items, sentence_id),
        )
        row = items.loc[items["sentence_id"] == selected_sentence].iloc[0]
        st.markdown(
            f"**{row['ticker']} {row['call_key'] or ''}** · "
            f"{row['call_date_et'] or 'date unknown'} · {row['section'] or 'section unknown'} · "
            f"Speaker: {row['speaker_type'] or 'unknown'}"
        )
        if row["double_label"]:
            st.info("This passage is in the 20% double-label subset.")
        st.markdown("**Passage and neighboring context**")
        st.write(row["context_text"])
        st.caption(f"Technology term(s): {row['ai_terms'] or 'not recorded'}")
        st.markdown(f"**Anchor sentence:** {row['anchor_sentence']}")

        if not annotator:
            st.warning("Enter your name or initials in the sidebar before saving a label.")
        existing = con.execute(
            "SELECT label, notes FROM annotations "
            "WHERE batch_id=? AND sentence_id=? AND annotator=?",
            (batch_id, selected_sentence, annotator),
        ).fetchone() if annotator else None
        prior_label = existing[0] if existing else None
        choice = st.radio(
            "Label",
            list(annotation.LABELS),
            index=list(annotation.LABELS).index(prior_label) if prior_label is not None else 0,
            format_func=lambda level: annotation.LABELS[level],
            horizontal=False,
        )
        notes = st.text_area("Notes (optional)", value=existing[1] or "" if existing else "")
        if st.button("Save label", type="primary", disabled=not annotator):
            annotation.save_annotation(
                con, batch_id, selected_sentence, annotator, int(choice), notes
            )
            st.success("Label saved.")
            st.rerun()

        st.divider()
        st.subheader("Inter-annotator agreement")
        raters = annotation.annotators_for_batch(con, batch_id)
        if len(raters) < 2:
            st.caption("Cohen’s kappa appears after two annotators have saved labels.")
        else:
            left, right = st.columns(2)
            annotator_a = left.selectbox("Annotator A", raters, key="kappa_a")
            other_raters = [name for name in raters if name != annotator_a]
            annotator_b = right.selectbox("Annotator B", other_raters, key="kappa_b")
            stats = annotation.cohen_kappa(con, batch_id, annotator_a, annotator_b)
            if stats["n"] == 0:
                st.info("The selected annotators have no jointly labeled double-label passages yet.")
            elif stats["kappa"] is None:
                st.metric("Cohen’s kappa", "Undefined")
                st.caption(
                    f"{stats['n']} passages; expected agreement is 1.0, so kappa’s denominator is zero."
                )
            else:
                st.metric("Cohen’s kappa", f"{stats['kappa']:.3f}")
                st.caption(
                    f"Based on {stats['n']} jointly labeled double-label passages "
                    f"({stats['observed_agreement']:.1%} observed agreement)."
                )
    finally:
        con.close()


def _passage_option(items, sentence_id: str) -> str:
    row = items.loc[items["sentence_id"] == sentence_id].iloc[0]
    status = "labeled" if row["annotation_count"] else "not labeled"
    double = " · double-label" if row["double_label"] else ""
    snippet = str(row["anchor_sentence"]).replace("\n", " ")[:75]
    return f"{row['sample_index']:03d} · {row['ticker']} · {status}{double} · {snippet}"


if __name__ == "__main__":
    main()
