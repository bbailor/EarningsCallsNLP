"""Step-by-step demo of the NLP results (reads existing outputs; re-runs nothing).

    python demo.py                 # walk through every step, Enter to advance
    python demo.py "Some sentence" # just score one sentence with FinBERT
"""
import sqlite3
import sys

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 110)
OUT = "data/processed"


def pause(title):
    input(f"\n[Enter] -> {title} ")
    print("=" * 80 + f"\n{title}\n" + "=" * 80)


def score(sentences):
    from ecnlp.sentiment import FinBertScorer, LABELS

    print("Loading FinBERT (ProsusAI/finbert)...")
    probs = FinBertScorer("ProsusAI/finbert")(sentences)
    for s, p in zip(sentences, probs):
        print(f"\n  \"{s}\"\n  -> {LABELS[p.argmax()].upper():8s}  pos={p[0]:.2f} neg={p[1]:.2f} "
              f"neu={p[2]:.2f}  score={p[0] - p[1]:+.2f}")


def live_scoring():
    score(["Revenue grew 22% to a record $35 billion, driven by strong AI demand.",
           "Margins declined due to higher tariffs and weaker consumer spending.",
           "We will hold our annual shareholder meeting in June."])
    while True:
        s = input("\nType a sentence to score (blank to continue): ").strip()
        if not s:
            break
        score([s])


def main():
    if len(sys.argv) > 1:
        score([" ".join(sys.argv[1:])])
        return

    calls = pd.read_csv(f"{OUT}/nlp_call_scores.csv")
    sents = pd.read_parquet(f"{OUT}/nlp_sentence_scores.parquet")
    topics = pd.read_csv(f"{OUT}/nlp_topics.csv")

    pause("1. The data: transcripts pulled and prepared")
    con = sqlite3.connect("data/earnings_nlp.sqlite")
    for table in ("calls", "segments", "sentences", "sentiment"):
        print(f"  {table:10s} {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>8,} rows")
    con.close()
    print(calls.groupby("ticker").agg(calls=("call_id", "size"), first=("call_date_et", "min"),
                                      last=("call_date_et", "max")).to_string())

    pause("2. FinBERT live: sentence -> sentiment")
    live_scoring()

    pause("3. What one sentence looks like after the pipeline")
    ex = sents[sents["has_ai_term"] == 1].sample(5, random_state=7)
    print(ex[["ticker", "call_key", "section", "speaker_type", "label", "score", "topic_id", "text"]]
          .round(2).to_string(index=False))

    pause("4. Call-level variables: average tone by company")
    cols = ["sent_all_mean", "sent_prepared_mean", "sent_qa_mean", "sent_ai_ctx_mean", "sent_ai_minus_nonai"]
    print(calls.groupby("ticker")[cols].mean().round(3).sort_values("sent_all_mean", ascending=False).to_string())
    print("\n  -> Prepared remarks are more positive than Q&A for every company.")
    print("  -> AI talk is more positive than the rest of the call, except AAPL and JPM.")

    pause("5. One company over time: NVDA")
    nv = calls[calls["ticker"] == "NVDA"].sort_values("call_date_et")
    print(nv[["call_key", "call_date_et", "sent_all_mean", "sent_qa_mean", "sent_ai_ctx_mean",
              "n_ai_sentences"]].round(3).to_string(index=False))

    pause("6. BERTopic: the largest topics")
    print(topics.sort_values("n_docs", ascending=False)[["topic_id", "n_docs", "top_words"]]
          .head(15).to_string(index=False))

    pause("7. Most positive / most negative sentences (and a FinBERT limitation)")
    for lab, asc in (("positive", False), ("negative", True)):
        print(f"\n  Most {lab}:")
        print(sents.sort_values("score", ascending=asc)[["ticker", "score", "text"]].head(4)
              .round(2).to_string(index=False))
    print('\n  Limitation: FinBERT reacts to direction words. "Operating expenses ... DOWN 10%" above is good news '
          "(costs cut)\n  but scores as strongly negative. Sentence scores need spot-checking.")
    print("\nDone.")


if __name__ == "__main__":
    main()
