# Project: Shapley Interactions on LLMs — Text Imputers & Cool Use Cases

**Type:** Pull Request(s) + Demo

## Overview

Language models are everywhere — encoder classifiers, causal LLMs for chat and code, seq2seq models for translation and summarization — and for every one of them, there is fascinating behavior that nobody fully understands. Where does a jailbreak actually "act"? Which few-shot demonstration did an in-context answer rely on? Which retrieved chunk is the RAG answer really grounded in? Which words in a prompt *interact* to flip a sentiment prediction? Shapley values and **Shapley interactions** are uniquely well-suited to answering this class of question, and shapiq has the full game-theoretic machinery — any interaction order, many indices (SII, k-SII, STII, FSII, FBII, BV, BII), many approximators — to go beyond what every other SHAP library offers.

This project has two sides: **building a text imputer** for shapiq (the PR side) and **exploring interesting use cases** that showcase what Shapley interactions can reveal about language model behavior (the Demo side). The core question driving both is:

> *What can shapiq show about LLM behavior that nothing else can — and what's the best way to enable it?*

The PR deliverable is a **text imputer** — a clean, tested component that makes text/LLM explanations a first-class capability in shapiq. Once you have that, you'll use it to build polished demos on interesting use cases (jailbreaks, RAG attribution, in-context learning, and more). Your team has freedom in choosing **which use cases to explore** and how deep to go — pick the demos that excite you most.

The starting points are shapiq's existing text handling: a hard-coded sentiment-analysis benchmark game (`shapiq_games.benchmark.local_xai.SentimentAnalysis`) and a sentence visualization utility (`src/shapiq/plot/sentence.py`).

## Tasks

### Task 1: Build a text imputer for shapiq (the PR)

This is the engineering heart of the project. Build a **text imputer** — a subclass of `shapiq.imputer.base.Imputer` that lets you mask player-defined spans of a prompt and call an arbitrary LLM on the masked versions. This is what you need to build:

- A flexible **player definition** — at minimum token-level and word-level players; span-level players (sentences, retrieved chunks, few-shot demonstrations) for more advanced use cases.
- Pluggable **masking strategies** — `[MASK]` replacement, `[PAD]` replacement, token removal, attention masking, and ideally at least one novel strategy (e.g. MLM-infill, neutral replacement).
- A flexible **target callable** — classification logits for encoder models, next-token / target-continuation log-likelihood for causal LLMs, and optionally perplexity, contrastive log-odds, etc.
- **Batched model calls** — LLMs are expensive; the value function must evaluate batches of coalitions efficiently.
- Integration with **HuggingFace `transformers`** (torch backend is sufficient; JAX/Flax is a nice-to-have).

Of course, if you want to go beyond the text imputer, you are welcome to contribute additional infrastructure as well — for example:

- **Text/LLM games:** New game classes (subclassing `shapiq.game.Game`, or as new `shapiq_games` benchmark games) tailored to LLM explanation scenarios — e.g. a prompt-attribution game, a RAG-grounding game, a jailbreak-detection game. Look at the existing `SentimentAnalysis` benchmark game for the pattern.
- **Visualization utilities:** Extensions to `src/shapiq/plot/sentence.py` — token-interaction heatmaps, interaction-graph overlays on prompts, side-by-side comparison plots for different indices or models.
- **Anything else** that would make text/LLM explanations in shapiq better for future users.

But the text imputer is the core deliverable. Whatever you build must include tests, docstrings, and pass pre-commit. The existing `SentimentAnalysis` benchmark game (`src/shapiq_games/benchmark/local_xai/benchmark_language.py`) and `src/shapiq/plot/sentence.py` are your starting points.

### Task 2: The demos — showcase your text imputer on interesting use cases

> **Note on scaffolding:** You will likely need some quick-and-dirty scaffolding code to unblock your demo exploration before the PR code is polished. That's expected — build the minimum viable version first, get to the interesting demos, and then decide what's clean enough to promote to the PR. Work with HuggingFace `transformers` (torch backend); prefer real SOTA models — smaller open-weight models (1–8B range) are fine if larger ones don't fit your hardware.

Now that you have a text imputer, use it to build **at least three** polished, self-contained demos that showcase what Shapley interactions can reveal (or cannot reveal) about language model behavior. Here are some directions we find interesting — you are welcome to replace these with better ideas:

- **Prompt-injection and jailbreaks.** Feed a model a benign prompt plus a jailbreak payload. Use Shapley interactions to show which token groups *interact* to cause the jailbreak — and which don't. Does the interaction structure distinguish real jailbreaks from innocuous phrasing?
- **In-context learning attribution.** For a few-shot prompt, attribute the answer back to individual demonstrations (treat each demonstration as one player). Which example did the model actually rely on? Are there interaction effects between demonstrations?
- **RAG / retrieval attribution.** For a retrieval-augmented answer, treat each retrieved chunk (or each sentence in each chunk) as a player. Identify the *source of groundedness* and flag unsupported answers where no chunk has a meaningful attribution.
- **Chain-of-thought attribution.** Attribute a final answer back to the model's own reasoning steps. Which step was load-bearing? Which was filler?
- **Contrastive / counterfactual explanations.** Given two almost-identical prompts with very different outputs (e.g. minor negation flip, pronoun swap), show the interactions responsible for the divergence.
- **Agentic tool-use explanations.** For a tool-calling agent, attribute the decision to call (or not call) a specific tool back to tokens in the user request and system prompt.
- **Multilinguality & robustness.** Compare attributions across translations or paraphrases of the same prompt — are explanations stable? Are the same interactions present?
- **Word-level interactions in classification.** On sentiment / NLI / toxicity tasks, use higher-order interactions to show phenomena first-order SVs miss: negation + adjective, subject-verb agreement, multi-word named entities, sarcasm cues.

For each demo:

1. **Frame the question clearly.** What are you trying to show? Why is it interesting?
2. **Pick a concrete SOTA model** (or a small set) and a concrete input (real jailbreak payloads from public datasets, real RAG traces, real few-shot prompts, etc.). Use genuinely interesting examples, not toy inputs.
3. **Use the right interaction index.** Shapley values alone are fine for simple cases, but the project's unique angle is interactions — use **k-SII, STII, or FSII** where pairwise or higher-order structure matters. Make a deliberate choice and explain it.
4. **Visualize the result.** Extend or reuse `src/shapiq/plot/sentence.py`; add heatmaps, interaction graphs, side-by-side comparisons — whatever makes the findings clearest. Visual quality matters.
5. **Draw a conclusion.** What did you learn? What's surprising? Where did the method break down? An honest "this didn't work and here's why" is a legitimate demo result.

Format the demos as a collection of notebooks, a Gradio / Streamlit app, or a Hugging Face Space — whatever serves the content best. They should be fully reproducible (fixed seeds, pinned HF model revisions, clear install instructions) and runnable by anyone with a reasonable GPU.

### Task 3: Comparison with existing libraries

A demo that only shows what shapiq can do isn't enough — we also want to know how it compares. Include at least one comparison section (a notebook, a page, a dashboard) where you run shapiq head-to-head against existing text-explanation libraries on a shared input + model:

- [`shap.Explainer` with `shap.maskers.Text`](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Text.html) — the current de-facto Shapley-on-text baseline.
- [captum's `ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html) applied at the token-embedding level.
- [**Inseq**](https://inseq.org/) — a dedicated sequence-attribution library for generation; especially relevant for causal LLM comparisons.

Report: (i) runtime and memory, (ii) agreement of attributions — do the same tokens come out as important? if not, why? (iii) API ergonomics, (iv) **what shapiq can uniquely do**: any-order interactions, many indices — show concrete examples where this buys something the baselines cannot offer.

Honest discussion beats a cherry-picked win: where shapiq is slower or clunkier, say so.

### Task 4: Additional PRs (optional)

If your demo exploration surfaces additional pieces of reusable code beyond your main PR (Task 1), you are encouraged to upstream them as additional PRs. This is a bonus, not a requirement — but well-motivated additions are always welcome. Each PR should meet shapiq's normal bar: tests, docstrings, pre-commit passes, and a clear motivation in the PR description.

## Relevant Existing Code

| Path | Description |
|------|-------------|
| `src/shapiq_games/benchmark/local_xai/benchmark_language.py` | Existing hard-coded `SentimentAnalysis` benchmark game (DistilBERT + `[MASK]` / removal) |
| `src/shapiq/plot/sentence.py` | Sentence-level visualization — primary starting point for token / interaction overlays |
| `docs/source/auto_examples/language/plot_sentiment_analysis.py` | Existing sentiment-analysis example (KernelSHAP + KernelSHAPIQ) |
| `src/shapiq/imputer/base.py` | `Imputer` base class — if you build scaffolding, subclass this |
| `src/shapiq/imputer/marginal_imputer.py` / `baseline_imputer.py` | Imputer references for shape and API conventions |
| `src/shapiq/explainer/tabular.py` | How an imputer plugs into an explainer |
| `src/shapiq/game_theory/exact.py` | `ExactComputer` — ground truth for any correctness test on small inputs |
| `src/shapiq/game_theory/indices.py` | Interaction indices available in shapiq (SV, SII, k-SII, STII, FSII, FBII, BV, BII) |

## References

- **SHAP text maskers:** *SHAP documentation on text maskers*. [shap.maskers.Text](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Text.html) — the de-facto standard for Shapley on text.
- **Inseq:** Sarti et al., *Inseq: An Interpretability Toolkit for Sequence Generation Models*, ACL 2023. [arXiv:2302.13942](https://arxiv.org/abs/2302.13942).
- **Captum:** Kokhlikyan et al., *Captum: A unified and generic model interpretability library for PyTorch*, 2020. [arXiv:2009.07896](https://arxiv.org/abs/2009.07896).
- **Ferret:** Attanasio et al., *ferret: a Framework for Benchmarking Explainers on Transformers*, EACL 2023. [arXiv:2208.01575](https://arxiv.org/abs/2208.01575) — methodology template for evaluating and comparing text-explanation methods.
- **Shapley attributions for LLMs — recent work:** survey the 2024–2026 arXiv literature on prompt attribution, jailbreak explanation, RAG attribution, and in-context-learning attribution. Bring a reading list to your first group meeting.
- **shapiq paper:** Muschalik et al., *shapiq: Shapley Interactions for Machine Learning*, NeurIPS 2024 — for the architecture and indices you are building on.

## Expected Deliverables

**PR(s):**

- A clean, well-tested PR contributing a text imputer to shapiq (see Task 1). Additional infrastructure contributions (games, visualization utilities) are welcome but not required.
- Tests, docstrings, and passing pre-commit for all PR code.
- All existing tests and pre-commit checks must continue to pass (`uv run pre-commit run --all-files`, `uv run pytest tests/shapiq`).

**Demo:**

- A polished, reproducible demo covering **at least three** distinct use cases of Shapley values / Shapley interactions on LLMs, on real SOTA models and real inputs.
- A dedicated comparison section against `shap`, `captum`, and `Inseq` on a shared input + model, with honest runtime / agreement / ergonomics reporting.
- Clear visualizations that make the findings legible to non-expert viewers.
- Fully reproducible: pinned dependencies, pinned HF model revisions, fixed seeds, explicit install and run instructions.
