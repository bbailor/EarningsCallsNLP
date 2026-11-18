# NLP Variable Definitions

Team GPT Gains. Formal definitions of every text-derived variable the pipeline produces, with the formula and the code that computes it. Market variables (returns, CARs) are not covered here.

Settings quoted below are the current values in `config.yaml`. If a setting changes, the variable changes with it, so cite the config value when reporting results.

## 1. Notation and units of analysis

| Symbol | Meaning |
| --- | --- |
| $c$ | an earnings call (`call_id`) |
| $s$ | a sentence (`sentence_id` = `<segment_id>-<sentence index>`) |
| $S_c$ | the analysis sentences of call $c$ (defined below) |
| $\mathbb{1}[\cdot]$ | indicator: 1 if the condition holds, else 0 |
| $\lvert A \rvert$ | number of elements in set $A$ |

**Segment (speaker turn).** One uninterrupted turn by one speaker, as returned by earningscalls.dev, with `speaker_type` in {executive, analyst, operator, shareholder, unknown}.

**Section.** Each turn is labeled `prepared_remarks` or `qa`. Q&A begins at the first analyst turn that comes after an executive has spoken (first shareholder turn if there are no analysts), and the operator turn just before it is also Q&A (`cleaning.assign_sections`).

**Sentence.** Cleaned turn text (Unicode-normalized, transcription notes removed) split with a rule-based splitter that handles abbreviations, decimals and initials (`cleaning.split_sentences`). Operator turns are dropped before splitting (`drop_operator_segments: true`).

**Word count.** $w(x)$ = number of tokens in text $x$ matching the regular expression `\b[\w'$%.-]+\b` (`cleaning.word_count`).

**Boilerplate.** $b(s) = 1$ if sentence $s$ matches the safe-harbor pattern (forward-looking statements, safe harbor, "actual results may differ materially", non-GAAP reconciliation, "this call is being recorded", Form 10-K/10-Q, etc.; `cleaning.is_boilerplate`).

**AI term flag.** $a(s) = 1$ if sentence $s$ contains at least one term from the AI keyword dictionary (`cleaning.build_ai_matcher`), matched on word boundaries:
- case-sensitive: `AI`, `A.I.`, `GenAI`/`Gen AI`, `ML`, `LLM(s)`, `GPT`(`-n`)
- case-insensitive: artificial intelligence, machine learning, generative ai, large language model(s), deep learning, neural network(s), agentic, ai agent(s), copilot(s), chatbot(s), computer vision, natural language processing, predictive analytics

The matched terms are stored pipe-separated in `ai_terms`.

**Analysis sentences.** Sentiment uses
$$S_c = \{\, s \in c : b(s) = 0,\ w(s) \ge 3 \,\}$$
(`nlp.sentiment.min_words: 3`). The topic model uses a stricter set with $w(s) \ge 8$ (`nlp.topics.min_words: 8`), written $S^{T}_c$, because short sentences make noisy topics.

**AI context (AI mention unit).** For each sentence $s$ with $a(s)=1$ and $b(s)=0$, the AI context $x_s$ is $s$ joined with up to $k$ sentences before and after it **from the same speaker turn**, with $k$ = `context_window` = 1 (`exports.build_ai_contexts`). Neighbors are not filtered for boilerplate or length. The context is keyed by the `sentence_id` of the AI sentence. $A_c$ denotes the set of AI contexts in call $c$.

## 2. Call-level text and AI-intensity variables

Source: `calls` table, `data/processed/calls_master.csv` and `nlp_call_scores.csv`.

| Variable | Definition |
| --- | --- |
| `mentions_ai` | 1 if the call matched the earningscalls.dev full-text search `ai_query` (`"artificial intelligence" OR AI OR "machine learning" OR "generative AI" OR GenAI OR LLM OR "large language model" OR agentic`), else 0. Set by the API at pull time, not by our keyword list. With `include_non_ai_calls: false`, every call in the current sample has `mentions_ai = 1`. |
| `n_sentences` | Number of sentences in non-operator turns, **including** boilerplate. |
| `n_words_ex_operator` | $\sum_{g} w(g)$ over all non-operator turns $g$ of the call. |
| `n_ai_sentences` | $\sum_{s \in c} a(s)\,(1 - b(s))$: non-boilerplate sentences with an AI term. Not filtered by length. |
| `ai_sentences_per_1000_words` | $\dfrac{\texttt{n\_ai\_sentences}}{\texttt{n\_words\_ex\_operator}} \times 1000$; missing if the denominator is 0. AI intensity, comparable across calls of different length. |

## 3. FinBERT sentiment

### 3.1 Model

Model: `ProsusAI/finbert` (BERT-base further trained on Financial PhraseBank), used as-is, with no fine-tuning on our data. Text is tokenized and truncated to 256 tokens (`max_length`). For a unit of text $x$ (a sentence or an AI context), FinBERT outputs logits $z_j(x)$ for $j \in \{\text{pos}, \text{neg}, \text{neu}\}$, converted with softmax:

$$p_j(x) = \frac{e^{z_j(x)}}{\sum_{j'} e^{z_{j'}(x)}}, \qquad p_{\text{pos}} + p_{\text{neg}} + p_{\text{neu}} = 1$$

### 3.2 Unit-level variables

Source: `sentiment` table; sentence rows also in `nlp_sentence_scores.parquet`.

| Variable | Definition |
| --- | --- |
| `p_positive`, `p_negative`, `p_neutral` | $p_{\text{pos}}(x)$, $p_{\text{neg}}(x)$, $p_{\text{neu}}(x)$ |
| `label` | $\ell(x) = \arg\max_j p_j(x)$, one of positive / negative / neutral |
| `score` | $\sigma(x) = p_{\text{pos}}(x) - p_{\text{neg}}(x) \in [-1, 1]$ |

Two unit types are scored: `sentence` (every $s \in S_c$) and `ai_context` (every $x \in A_c$).

### 3.3 Call-level sentiment variables

Source: `nlp_call_scores.csv`. Let $U$ be a set of scored units in call $c$ (a "slice", defined below). With $N = \lvert U \rvert$, $N^{+} = \sum_{x \in U} \mathbb{1}[\ell(x) = \text{positive}]$ and $N^{-} = \sum_{x \in U} \mathbb{1}[\ell(x) = \text{negative}]$:

| Variable | Formula | Range |
| --- | --- | --- |
| `sent_<slice>_n` | $N$ | count |
| `sent_<slice>_mean` | $\dfrac{1}{N} \sum_{x \in U} \sigma(x)$ | $[-1, 1]$ |
| `sent_<slice>_net_tone` | $\dfrac{N^{+} - N^{-}}{N^{+} + N^{-}}$, missing if $N^{+} + N^{-} = 0$ | $[-1, 1]$ |
| `sent_<slice>_pct_pos` | $N^{+} / N$ (denominator includes neutral units) | $[0, 1]$ |
| `sent_<slice>_pct_neg` | $N^{-} / N$ | $[0, 1]$ |

`_mean` uses the probabilities, so it reflects confidence. `_net_tone` uses only the hard labels and ignores neutral sentences.

**Slices.** All sentence slices are subsets of $S_c$:

| Slice | Units $U$ |
| --- | --- |
| `all` | $S_c$ |
| `prepared` | $s \in S_c$ with section = prepared_remarks |
| `qa` | $s \in S_c$ with section = qa |
| `exec` | $s \in S_c$ spoken by an executive |
| `analyst` | $s \in S_c$ spoken by an analyst |
| `nonai` | $s \in S_c$ with $a(s) = 0$ |
| `ai_ctx` | AI contexts $A_c$ (unit type `ai_context`, not sentences) |

Note that `nonai` can include sentences that sit next to an AI mention and therefore also appear inside an AI context.

**AI tone gap:**
$$\texttt{sent\_ai\_minus\_nonai}_c = \texttt{sent\_ai\_ctx\_mean}_c - \texttt{sent\_nonai\_mean}_c$$
Positive values mean AI talk is more upbeat than the rest of the call. This differences out company- and call-wide tone, so it is the main sentiment variable for the AI question.

## 4. BERTopic topics

### 4.1 Model

One model is fit across all calls in the configured sample (`corpus: sentences`, documents = $\bigcup_c S^{T}_c$), so a topic ID means the same thing for every company and quarter. It is refit from scratch on every run, so topic IDs are not comparable across runs.

1. **Embedding.** Each sentence $s$ is mapped to $e(s) \in \mathbb{R}^{384}$ by `sentence-transformers/all-MiniLM-L6-v2`.
2. **Dimension reduction.** UMAP to 5 dimensions (cosine metric, `n_neighbors = 15`, `min_dist = 0`, seed 42).
3. **Clustering.** HDBSCAN (`min_cluster_size = 150`, excess-of-mass selection). Each cluster is a topic $k \in \{0, 1, \dots, K-1\}$; points in no cluster get $k = -1$ (outlier). With `reduce_outliers: true`, outliers are reassigned to the topic with the most similar embedding, so $k = -1$ is usually rare or absent.
4. **Topic words (class-based TF-IDF).** All sentences in topic $k$ are pooled into one document. For term $t$ (unigrams and bigrams, English plus earnings-call stop words removed):
$$W_{t,k} = \mathrm{tf}_{t,k} \cdot \log\!\left(1 + \frac{\bar{A}}{f_t}\right)$$
where $\mathrm{tf}_{t,k}$ is the frequency of $t$ in topic $k$, $f_t$ is the frequency of $t$ across all topics, and $\bar{A}$ is the average number of words per topic. The 10 terms with the highest $W_{t,k}$ are the topic's top words.

### 4.2 Topic-level and sentence-level variables

Source: `topics` and `topic_assignments` tables; `nlp_topics.csv`.

| Variable | Definition |
| --- | --- |
| `topic_id` | $k(s)$, the topic assigned to sentence $s$ |
| `topic_prob` / `prob` | HDBSCAN membership probability of $s$ in its topic; missing for sentences reassigned from outlier |
| `name` | BERTopic's label: topic ID plus top words |
| `top_words` | top 10 terms by $W_{t,k}$, pipe-separated |
| `n_docs` | number of sentences assigned to topic $k$ |
| `representative_doc` | an example sentence BERTopic picks as typical of the topic |

### 4.3 Call-level topic variables

Source: `nlp_call_scores.csv` (wide) and `nlp_topic_shares.csv` (long, every topic). Let $n_{c,k} = \lvert\{ s \in S^{T}_c : k(s) = k \}\rvert$, including $k = -1$.

| Variable | Formula / definition |
| --- | --- |
| `share` (long file) and `topic_<kk>_share` | $\pi_{c,k} = \dfrac{n_{c,k}}{\sum_{j} n_{c,j}}$, where the sum includes outliers. Wide columns exist for the 20 topics with the largest total $\sum_c n_{c,k}$ (`wide_top_n: 20`); 0 if the call has no sentences in that topic. |
| `topic_outlier_share` | $\pi_{c,-1}$; 0 if the call has no outlier sentences |
| `topic_n_distinct` | $\lvert\{ k \ge 0 : n_{c,k} > 0 \}\rvert$ |
| `topic_dominant` | $\arg\max_{k \ge 0} n_{c,k}$ (ties go to the lowest topic ID) |
| `topic_entropy` | normalized Shannon entropy, defined below |

**Topic entropy.** Using shares that exclude outliers,
$$\tilde{\pi}_{c,k} = \frac{n_{c,k}}{\sum_{j \ge 0} n_{c,j}}, \qquad H_c = -\frac{1}{\ln K} \sum_{k \ge 0,\ \tilde{\pi}_{c,k} > 0} \tilde{\pi}_{c,k} \ln \tilde{\pi}_{c,k}$$
where $K$ is the number of distinct non-outlier topics assigned anywhere in the sample ($H_c = 0$ if $K \le 1$). $H_c \in [0, 1]$: 0 means the whole call is one topic, and 1 means the call is spread evenly over all $K$ topics. Normalizing by the model's $K$ rather than the topics a call uses makes a 50/50 split over 2 topics score lower than an even spread over 40.

## 5. Things to keep in mind when using these variables

- **Different sentence sets.** Sentiment uses sentences with 3+ words; topics use 8+ words. Shares and sentiment for the same call are not over identical sentences.
- **Keyword vs API flag.** `mentions_ai` comes from the API's search; `n_ai_sentences` comes from our dictionary. A call can have `mentions_ai = 1` and `n_ai_sentences = 0` (the `prepare` stage logs these).
- **Topic IDs are per fit.** Re-running `topics` on a different sample renumbers topics. Record the run date with any topic result.
- **FinBERT measures tone, not substance.** It was not trained on AI language. Specific vs vague AI talk is the job of our own model (planned for Sprint 3, variables not defined yet).
- **Missing values.** A `sent_<slice>_*` variable is missing when the call has no units in that slice (for example, a call with no AI contexts). Topic variables are missing for calls with no sentences of 8+ words.

## 6. Where each variable is computed

| Variables | Code |
| --- | --- |
| Sections, sentences, boilerplate, AI flags, word counts | `src/ecnlp/cleaning.py`, `src/ecnlp/prepare.py`, `src/ecnlp/transcripts.py` |
| AI contexts | `src/ecnlp/exports.py` (`build_ai_contexts`) |
| FinBERT unit scores | `src/ecnlp/sentiment.py` |
| BERTopic fit and assignments | `src/ecnlp/topics.py` |
| Call-level sentiment and topic variables, NLP exports | `src/ecnlp/nlp_scores.py` |
