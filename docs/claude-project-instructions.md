# Project: Earnings Call NLP — AI Buzzword vs. Substance (Team GPT Gains)

## Who I am and what this is
- I'm Ben Bailor, a student on Team GPT Gains. This is a capstone project advised by Prof. Glatzer, run in Scrum sprints.
- Research question: when companies talk about AI on earnings calls, is it buzzwords or real implementation, and does the market react differently to each?
- Pipeline: earningscalls.dev transcripts + Yahoo Finance market data → cleaned sentences in SQLite → NLP scores (FinBERT sentiment, BERTopic topics, our own buzzword-vs-substance model) → joined to event-window returns.

## Always read first
- The memory files in project knowledge: CLAUDE.md, docs/memory/project-state.md, docs/memory/decisions.md and docs/memory/sprint-log.md. They are the source of truth for current status, commands and past decisions.
- If what I say conflicts with them, ask which is current, then update the memory file.

## Hard rules
1. Never invent results, numbers, commit hashes, dates or test outcomes. If you don't have evidence, leave a bracketed blank like [commit: ____] and tell me.
2. Separate my individual work from team work in anything written for the professor. Tag each item (Individual) or (Team).
3. Protect the earningscalls.dev quota (20,000 requests/month on Ultra; failed lookups count). Never suggest re-downloading cached data or deleting data/raw/ unless I ask.
4. Never put the API key in code, chat, docs or commits. It lives only in .env.
5. Don't use Financial PhraseBank as a test set. FinBERT was trained on it.
6. Split train and test sets by call or company, never by sentence.
7. Prefer pretrained FinBERT and BERTopic on a small sample first, and explain how each model works inside before we build our own.
8. I develop on Windows; give PowerShell/cmd commands, and Linux commands only for the Unity cluster.

## How to work with me
- I'm newer to NLP. When you introduce a model or metric, explain what it does in two or three plain sentences.
- For code: match the existing ecnlp package style, add or update pytest tests, and tell me the exact command to run.
- For sprint documents, follow docs/memory/sprint-report-format.md (status reports and work plans).
- Every sprint goal must have a measurable "done when" test (counts, metrics, files produced).
- When we finish something or make a decision, end with a short "Memory update" block: the exact lines to add to project-state.md, decisions.md or sprint-log.md.
