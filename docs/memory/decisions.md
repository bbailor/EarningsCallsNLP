# Decisions log

Newest first. Each entry: date, decision, why. Never delete entries; mark replaced ones "Superseded by <date>".

## 2026-11-11: Pretrained models first, then our own
Run pretrained FinBERT and BERTopic on a small sample of AI mentions, write one page on how each works, then build our own model. Why: Prof. Glatzer's Sprint 1 feedback, given our limited NLP experience.

## 2026-11-11: Status reports show evidence and individual credit
Every completed item cites a commit, output file or test result, and is tagged (Individual) or (Team). Every planned item has a measurable "done when". Why: Sprint 1 feedback.

## 2026-11-11: Keep non-AI calls as a comparison group
`include_non_ai_calls: true`, so every call has a yes/no AI flag. Why: the regressions need calls without AI talk. Cost: more API requests.

## 2026-11-02: Sample is the top 10 S&P 500 companies over five years
About 200 calls. Goal: summary statistics for at least one NLP variable (FinBERT sentiment plus word counts). Why: class meeting on 11/2; team estimated about two weeks.

## 2026-10-28: Build our own model and compare it with FinBERT and BERTopic
Evaluate on our own hand-labeled earnings-call test set (300-500 mentions, about 100 double-labeled), split by call. Why: Prof. Glatzer's requirement. Financial PhraseBank is excluded as a test set because FinBERT trained on it.

## 2026-10-28: Yahoo Finance instead of Capital IQ
Why: Capital IQ mass export needs a premium plug-in. The SDS was updated to match.

## 2026-10-28: Branch is master
The repo uses `master`, not `main`.
