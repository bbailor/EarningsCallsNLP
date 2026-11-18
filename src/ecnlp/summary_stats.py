"""NLP-07: summary statistics and figures for the baseline NLP variables.

Reads `calls_master.csv` (run `export` after `nlp` so it carries the NLP
scores), keeps the calls in config.yaml's tickers and dates that have FinBERT
scores, and writes to <data_dir>/processed/summary/:

    summary_overall.csv       N, mean, sd, min, quartiles, max per variable
    summary_by_ticker.csv     calls and variable means per company
    summary_by_year.csv       calls and variable means per calendar year of the call
    correlations_with_car.csv Pearson and Spearman correlation of each NLP variable
                              with the event-window returns (descriptive only:
                              calls from the same company are not independent)
    fig1..fig4 *.png          see FIGURES

Years are calendar years of `call_date_et`, not fiscal years: fiscal years
differ by company (NVDA's ends in January).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

# variable -> label used in tables and figures
NLP_VARS = {
    "n_words_ex_operator": "Words per call (non-operator)",
    "n_sentences": "Sentences per call",
    "n_ai_sentences": "AI sentences per call",
    "ai_sentences_per_1000_words": "AI sentences per 1,000 words",
    "sent_all_mean": "Sentiment, all sentences",
    "sent_all_net_tone": "Net tone, all sentences",
    "sent_prepared_mean": "Sentiment, prepared remarks",
    "sent_qa_mean": "Sentiment, Q&A",
    "sent_exec_mean": "Sentiment, executives",
    "sent_analyst_mean": "Sentiment, analysts",
    "sent_ai_ctx_mean": "Sentiment, AI mentions (with context)",
    "sent_nonai_mean": "Sentiment, non-AI sentences",
    "sent_ai_minus_nonai": "AI minus non-AI sentiment",
    "topic_entropy": "Topic entropy (0-1)",
    "topic_n_distinct": "Distinct topics per call",
}
RETURN_VARS = {
    "car_mkt_0_1": "Market-adjusted return, days 0 to +1",
    "car_mm_0_1": "Market-model abnormal return, days 0 to +1",
    "car_mkt_0_3": "Market-adjusted return, days 0 to +3",
}
BY_GROUP_VARS = ["n_words_ex_operator", "ai_sentences_per_1000_words", "sent_all_mean",
                 "sent_prepared_mean", "sent_qa_mean", "sent_ai_ctx_mean", "sent_nonai_mean",
                 "sent_ai_minus_nonai", "topic_entropy", "car_mkt_0_1"]

# Reference categorical palette (dataviz skill), validated light-mode for slots 1-2
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_sample(cfg: Config, master_path: Path | None = None) -> pd.DataFrame:
    """Calls in the configured sample that have FinBERT scores, with a calendar `year`."""
    path = master_path or cfg.paths.processed / "calls_master.csv"
    m = pd.read_csv(path)
    if "sent_all_mean" not in m.columns:
        raise ValueError(f"{path} has no NLP columns; run `nlp` then `export` first")
    d = m["call_date_et"].fillna("")
    keep = m["ticker"].isin(cfg.tickers) & (d >= cfg.date_from) & (d <= cfg.date_to) & m["sent_all_mean"].notna()
    dropped = m.loc[~keep, "ticker"].value_counts().to_dict()
    if dropped:
        log.info("Summary: dropped calls outside the scored sample: %s", dropped)
    s = m.loc[keep].copy()
    s["call_date"] = pd.to_datetime(s["call_date_et"])
    s["year"] = s["call_date"].dt.year
    return s.sort_values(["ticker", "call_date"]).reset_index(drop=True)


def describe(df: pd.DataFrame, variables: dict[str, str]) -> pd.DataFrame:
    rows = []
    for col, label in variables.items():
        if col not in df:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({"variable": col, "label": label, "n": len(x), "mean": x.mean(), "sd": x.std(),
                     "min": x.min(), "p25": x.quantile(0.25), "median": x.median(),
                     "p75": x.quantile(0.75), "max": x.max()})
    return pd.DataFrame(rows)


def by_group(df: pd.DataFrame, key: str) -> pd.DataFrame:
    cols = [c for c in BY_GROUP_VARS if c in df]
    g = df.groupby(key)
    out = g[cols].mean().add_prefix("mean_")
    out.insert(0, "n_calls", g.size())
    if key == "ticker":
        out.insert(1, "first_call", g["call_date_et"].min())
        out.insert(2, "last_call", g["call_date_et"].max())
    return out.reset_index()


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ret, ret_label in RETURN_VARS.items():
        if ret not in df:
            continue
        for col, label in NLP_VARS.items():
            if col not in df:
                continue
            pair = df[[col, ret]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3 or pair[col].nunique() < 2:
                continue
            rows.append({"nlp_variable": col, "nlp_label": label, "return": ret, "return_label": ret_label,
                         "n": len(pair), "pearson": pair[col].corr(pair[ret]),
                         "spearman": pair[col].corr(pair[ret], method="spearman")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def _style(ax, title: str | None = None, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8)


def fig_ai_intensity(df: pd.DataFrame, path: Path) -> Path:
    """Small multiples: AI sentences per 1,000 words over time, one panel per company."""
    import matplotlib.pyplot as plt

    tickers = sorted(df["ticker"].unique())
    ncol = 3
    nrow = int(np.ceil(len(tickers) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.3 * nrow), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, t in zip(axes.flat, tickers):
        d = df[df["ticker"] == t]
        ax.plot(d["call_date"], d["ai_sentences_per_1000_words"], color=BLUE, linewidth=2)
        ax.plot(d["call_date"], d["ai_sentences_per_1000_words"], "o", color=BLUE, markersize=3,
                markeredgecolor=SURFACE, markeredgewidth=1)
        _style(ax, t)
    for ax in axes.flat[len(tickers):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("AI sentences / 1,000 words", color=INK_2, fontsize=8)
    fig.suptitle("AI talk per call over time, by company", color=INK, fontsize=12, x=0.01, ha="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_ai_vs_nonai_tone(df: pd.DataFrame, path: Path) -> Path:
    """Dumbbell: mean FinBERT sentiment of AI mentions vs the rest of the call, per company."""
    import matplotlib.pyplot as plt

    g = df.groupby("ticker")[["sent_ai_ctx_mean", "sent_nonai_mean"]].mean().sort_values("sent_ai_ctx_mean")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(g) + 1.4), facecolor=SURFACE)
    ax.hlines(y, g["sent_nonai_mean"], g["sent_ai_ctx_mean"], color=GRID, linewidth=2, zorder=1)
    ax.scatter(g["sent_nonai_mean"], y, s=64, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=2,
               label="Non-AI sentences")
    ax.scatter(g["sent_ai_ctx_mean"], y, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="AI mentions (with context)")
    ax.set_yticks(y, g.index)
    _style(ax, "AI mentions are scored more positively than the rest of the call"
           if (g["sent_ai_ctx_mean"] > g["sent_nonai_mean"]).all() else
           "FinBERT sentiment: AI mentions vs the rest of the call",
           xlabel="Mean FinBERT sentiment, p(positive) - p(negative), per call then averaged")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_tone_by_year(df: pd.DataFrame, path: Path, min_calls: int = 5) -> Path:
    """Mean sentiment per calendar year, prepared remarks vs Q&A, with calls per year.

    Years with fewer than `min_calls` calls (e.g. a partial first year) are left
    out of the figure; summary_by_year.csv still has them.
    """
    import matplotlib.pyplot as plt

    g = df.groupby("year").agg(prep=("sent_prepared_mean", "mean"), qa=("sent_qa_mean", "mean"),
                               n=("call_id", "size"))
    dropped = g.index[g["n"] < min_calls].tolist()
    g = g[g["n"] >= min_calls]
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    for col, color, label in (("prep", BLUE, "Prepared remarks"), ("qa", ORANGE, "Q&A")):
        ax.plot(g.index, g[col], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(label, (g.index[-1], g[col].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.set_xticks(g.index, [f"{y}\n(n={n})" for y, n in zip(g.index, g["n"])])
    ax.set_xlim(g.index.min() - 0.3, g.index.max() + 0.9)
    _style(ax, "Prepared remarks are more positive than Q&A" if (g["prep"] > g["qa"]).all()
           else "FinBERT sentiment by year: prepared remarks vs Q&A",
           ylabel="Mean sentiment per call",
           xlabel=f"Calendar year of call (calls); {', '.join(map(str, dropped))} left out, < {min_calls} calls"
           if dropped else "Calendar year of call (calls)")
    ax.legend(["Prepared remarks", "Q&A"], frameon=False, fontsize=8, ncol=2, labelcolor=INK_2,
              loc="lower left", bbox_to_anchor=(0, 1.08), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_tone_vs_return(df: pd.DataFrame, path: Path, x: str = "sent_ai_minus_nonai",
                       y: str = "car_mkt_0_1") -> Path:
    """Scatter of an NLP variable against the 2-day market-adjusted return."""
    import matplotlib.pyplot as plt

    d = df[[x, y]].dropna()
    rho = d[x].corr(d[y], method="spearman")
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.scatter(d[x], d[y] * 100, s=36, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=1)
    _style(ax, f"AI tone vs 2-day market reaction (Spearman rho = {rho:.2f}, n = {len(d)})",
           ylabel="Market-adjusted return, days 0 to +1 (%)", xlabel=NLP_VARS.get(x, x))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


FIGURES = {
    "fig1_ai_intensity_by_company.png": fig_ai_intensity,
    "fig2_ai_vs_nonai_sentiment.png": fig_ai_vs_nonai_tone,
    "fig3_sentiment_by_year.png": fig_tone_by_year,
    "fig4_ai_tone_vs_return.png": fig_tone_vs_return,
}


def build_summary(cfg: Config, master_path: Path | None = None, figures: bool = True) -> dict[str, Any]:
    df = load_sample(cfg, master_path)
    out = cfg.paths.processed / "summary"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary_overall.csv": describe(df, {**NLP_VARS, **RETURN_VARS}),
        "summary_by_ticker.csv": by_group(df, "ticker"),
        "summary_by_year.csv": by_group(df, "year"),
        "correlations_with_car.csv": correlations(df),
    }
    files = {}
    for name, t in tables.items():
        t.to_csv(out / name, index=False, float_format="%.6g")
        files[name] = str(out / name)
    if figures:
        import matplotlib

        matplotlib.use("Agg")
        for name, fn in FIGURES.items():
            files[name] = str(fn(df, out / name))
    for k, v in files.items():
        log.info("summary %-30s -> %s", k, v)
    return {"calls": len(df), "tickers": int(df["ticker"].nunique()),
            "first_call": df["call_date_et"].min(), "last_call": df["call_date_et"].max(), "files": len(files)}
