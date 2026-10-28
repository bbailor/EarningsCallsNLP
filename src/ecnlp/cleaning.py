"""Text cleaning and segmentation that prepares transcripts for FinBERT/BERTopic.

Kept dependency-free (regex only) per the SDS "regex fallback" decision, so
it runs anywhere. spaCy can replace `split_sentences` later without changing
the table layout; sentence IDs are positional and stay stable as long as the
splitter is unchanged.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable

# ----------------------------------------------------------------- normalize
_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "…": "...", " ": " ",
    "​": "", "﻿": "",
}
_BRACKET_NOTES = re.compile(
    r"\[(?:operator instructions|inaudible|indiscernible|crosstalk|technical difficulty|"
    r"foreign language|audio gap|pause|laughter|music)[^\]]*\]",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Unicode-normalize, unify quotes/dashes, drop transcription notes, squeeze whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    for a, b in _REPLACEMENTS.items():
        t = t.replace(a, b)
    t = _BRACKET_NOTES.sub(" ", t)
    t = "".join(ch for ch in t if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return _WS.sub(" ", t).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'$%.-]+\b", text)) if text else 0


# ------------------------------------------------------------------ sections
def assign_sections(speaker_types: list[str | None]) -> list[str]:
    """Label each turn prepared_remarks or qa.

    Same rule the API documents for its `section` filter: Q&A starts at the
    first analyst turn (first shareholder turn if there are no analysts).
    Computing it locally saves one API request per call.
    """
    types = [(s or "").lower() for s in speaker_types]
    start = next((i for i, s in enumerate(types) if s == "analyst"), None)
    if start is None:
        start = next((i for i, s in enumerate(types) if s == "shareholder"), None)
    if start is None:
        return ["prepared_remarks"] * len(types)
    # The operator line that introduces the first question belongs to Q&A.
    if start > 0 and types[start - 1] == "operator":
        start -= 1
    return ["prepared_remarks" if i < start else "qa" for i in range(len(types))]


# ----------------------------------------------------------------- sentences
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "corp", "co", "ltd", "llc",
    "plc", "vs", "etc", "e.g", "i.e", "approx", "est", "no", "nos", "fig", "dept", "u.s",
    "u.k", "e.u", "a.i", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "q", "yr", "yrs", "avg", "mgmt", "govt", "intl", "mt", "ft",
}
_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9$])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter tuned for transcripts.

    Handles abbreviations (Mr., Inc., U.S., A.I.), decimals ($1.5 billion) and
    initials: a split is undone when the text before it ends in a known
    abbreviation or a single capital initial (e.g. "John F. Kennedy").
    """
    if not text:
        return []
    out: list[str] = []
    for piece in (p.strip() for p in _SPLIT.split(text)):
        if not piece:
            continue
        if out:
            prev = out[-1]
            tokens = prev.rstrip("\"')]").split()
            last = tokens[-1] if tokens else ""
            last_l = last.lower().rstrip(".")
            if last.endswith(".") and (last_l in _ABBREVIATIONS or (len(last_l) == 1 and last_l.isalpha())):
                out[-1] = f"{prev} {piece}"
                continue
        out.append(piece)
    return out


# ------------------------------------------------------- boilerplate / safe-harbor
_BOILERPLATE = re.compile(
    r"forward[- ]looking statements?|safe harbor|private securities litigation reform act|"
    r"actual results (?:may|could) differ materially|risk factors (?:described|discussed|included)|"
    r"undertake no (?:obligation|duty) to update|non-gaap (?:financial )?measures? .{0,60}reconciliation|"
    r"reconciliation .{0,60}non-gaap|this call is being recorded|webcast .{0,40}replay|"
    r"(?:form|forms) 10-[kq]",
    re.IGNORECASE,
)


def is_boilerplate(sentence: str) -> bool:
    return bool(_BOILERPLATE.search(sentence))


# ------------------------------------------------------------ AI keyword flag
def build_ai_matcher(case_sensitive: Iterable[str], case_insensitive: Iterable[str]) -> Callable[[str], list[str]]:
    """Compile the configured keyword list into a function text -> matched terms.

    Word boundaries are lookarounds (not \\b) so terms ending in '.' such as
    'A.I.' still match. Possessives/plurals like "AI's" match the base term.
    """
    def alt(terms: Iterable[str]) -> str:
        terms = [t for t in terms if t]
        return "|".join(sorted(terms, key=len, reverse=True))

    pats = []
    cs = alt(case_sensitive)
    if cs:
        pats.append(re.compile(rf"(?<!\w)(?:{cs})(?!\w)"))
    ci = alt(case_insensitive)
    if ci:
        pats.append(re.compile(rf"(?<!\w)(?:{ci})(?!\w)", re.IGNORECASE))

    def match(text: str) -> list[str]:
        found: list[str] = []
        for p in pats:
            for m in p.finditer(text or ""):
                term = m.group(0)
                norm = term if p.flags & re.IGNORECASE == 0 else term.lower()
                if norm not in found:
                    found.append(norm)
        return found

    return match


# --------------------------------------------------------------- fiscal period
_Q_PATTERNS = [
    re.compile(r"\bQ([1-4])\s*(?:FY\s*)?'?(\d{2,4})\b", re.IGNORECASE),        # Q3 2025 / Q3 FY25
    re.compile(r"\b(?:FY\s*)?'?(\d{4})\s*Q([1-4])\b", re.IGNORECASE),          # 2025 Q3
    re.compile(r"\b(first|second|third|fourth)\s+quarter\s+(?:fiscal\s+(?:year\s+)?)?(\d{4})", re.IGNORECASE),
]
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def parse_fiscal_period(title: str | None) -> tuple[int | None, int | None]:
    """Fallback when the API's fiscal_year/fiscal_quarter fields are null."""
    if not title:
        return None, None
    m = _Q_PATTERNS[0].search(title)
    if m:
        return _year(m.group(2)), int(m.group(1))
    m = _Q_PATTERNS[1].search(title)
    if m:
        return _year(m.group(1)), int(m.group(2))
    m = _Q_PATTERNS[2].search(title)
    if m:
        return int(m.group(2)), _ORD[m.group(1).lower()]
    return None, None


def _year(s: str) -> int:
    y = int(s)
    return 2000 + y if y < 100 else y
