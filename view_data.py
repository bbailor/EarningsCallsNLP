import sqlite3
import pandas as pd

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 200)

print("\n=== Coverage ===")
print(pd.read_csv("data/processed/coverage_report.csv").to_string(index=False))

print("\n=== Calls ===")
calls = pd.read_csv("data/processed/calls_master.csv")
print(calls[["ticker", "call_key", "call_date_et", "call_session", "day0",
             "n_ai_sentences", "car_mkt_0_1", "surprise_pct"]].to_string(index=False))

print("\n=== Sample AI mentions ===")
ctx = pd.read_parquet("data/processed/nlp_ai_contexts.parquet")
print(ctx[["ticker", "call_key", "ai_terms", "context_text"]].head(10).to_string(index=False))

print("\n=== Run log ===")
con = sqlite3.connect("data/earnings_nlp.sqlite")
print(pd.read_sql("SELECT stage, status, api_requests, cache_hits FROM run_log", con).to_string(index=False))
con.close()