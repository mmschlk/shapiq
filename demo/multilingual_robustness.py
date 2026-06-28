"""Multilingual robustness demo with Gemma and TextImputer.

This demo compares translations of the same sentiment meaning. Each phrase is
one span-level player. The value function is:

    v(S) = log P_Gemma(target_label | selected phrase spans in S)

The target label is "negative" because all examples express negative sentiment.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import shapiq
from shapiq.imputer.text_imputer import TextImputer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "google/gemma-4-E2B-it"
INTERACTION_INDEX = "k-SII"
INTERACTION_ORDER = 2
LOCAL_FILES_ONLY = True
DEVICE = "cpu"


class MultilingualCase(TypedDict):
    """One translated sentiment example."""

    language: str
    target_label: str
    phrase_names: list[str]
    phrase_kinds: list[str]
    phrase_spans: list[str]


CASES: list[MultilingualCase] = [
    {
        "language": "english",
        "target_label": "negative",
        "phrase_names": ["subject", "negation", "positive word"],
        "phrase_kinds": ["subject", "negation", "sentiment"],
        "phrase_spans": ["The movie", "was not", "good"],
    },
    {
        "language": "chinese",
        "target_label": "negative",
        "phrase_names": ["subject", "negation", "positive word"],
        "phrase_kinds": ["subject", "negation", "sentiment"],
        "phrase_spans": ["\u8fd9\u90e8\u7535\u5f71", "\u5e76\u4e0d", "\u597d"],
    },
    {
        "language": "german",
        "target_label": "negative",
        "phrase_names": ["subject", "negation", "positive word"],
        "phrase_kinds": ["subject", "negation", "sentiment"],
        "phrase_spans": ["Der Film", "war nicht", "gut"],
    },
]

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
PROMPT_TEMPLATE = "{text}"


def build_text(case: MultilingualCase) -> str:
    """Build the full prompt from phrase spans."""
    sentence = " ".join(case["phrase_spans"])
    return (
        "Classify the sentiment of this sentence as positive or negative.\n"
        f"Sentence: {sentence}\n"
        "Sentiment:"
    )


class PhrasePlayerStrategy:
    """Treat phrase spans as players while keeping the task instruction fixed."""

    def __init__(self, case: MultilingualCase) -> None:
        self.case = case
        self.phrase_spans = case["phrase_spans"]

    def get_players(self) -> list[str]:
        return self.phrase_spans

    @property
    def n_players(self) -> int:
        return len(self.phrase_spans)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: object,  # noqa: ARG002
    ) -> str:
        """Build a prompt from selected phrase spans; missing spans are removed."""
        if len(coalition) != self.n_players:
            msg = (
                f"Coalition length {len(coalition)} does not match "
                f"n_players={self.n_players}"
            )
            raise ValueError(msg)

        selected_sentence = " ".join(
            phrase
            for keep, phrase in zip(coalition, self.phrase_spans, strict=False)
            if keep
        )

        return (
            "Classify the sentiment of this sentence as positive or negative.\n"
            f"Sentence: {selected_sentence}\n"
            "Sentiment:"
        )


def all_coalitions(n_players: int) -> np.ndarray:
    """Enumerate all binary coalitions."""
    return np.array(
        list(itertools.product([0, 1], repeat=n_players)),
        dtype=bool,
    )


def _score_lookup(coalitions: np.ndarray, scores: np.ndarray) -> dict[tuple[int, ...], float]:
    """Create a lookup table from coalition to value-function score."""
    return {
        tuple(coalition.astype(int)): float(score)
        for coalition, score in zip(coalitions, scores, strict=False)
    }


def estimate_single_phrase_effects(
    coalitions: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Estimate v({i}) - v({}) for each phrase."""
    n_players = coalitions.shape[1]
    lookup = _score_lookup(coalitions, scores)

    empty = tuple([0] * n_players)
    empty_score = lookup[empty]

    effects = np.zeros(n_players, dtype=float)
    for i in range(n_players):
        only_i = [0] * n_players
        only_i[i] = 1
        effects[i] = lookup[tuple(only_i)] - empty_score

    return effects


def compute_shapiq_pairwise_interactions(
    imputer: TextImputer,
) -> tuple[np.ndarray, shapiq.InteractionValues]:
    """Compute formal second-order interaction values with shapiq."""
    computer = shapiq.ExactComputer(
        n_players=imputer.n_players,
        game=imputer,
    )
    interaction_values = computer(
        index=INTERACTION_INDEX,
        order=INTERACTION_ORDER,
    )
    interaction_matrix = interaction_values.get_n_order_values(order=INTERACTION_ORDER)
    return interaction_matrix, interaction_values


def plot_single_phrase_effects(
    case: MultilingualCase,
    effects: np.ndarray,
) -> Path:
    """Plot and save individual phrase effects."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"multilingual_{case['language']}_single_phrase_effects.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:red" if value > 0 else "tab:blue" for value in effects]
    ax.bar(case["phrase_names"], effects, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Single phrase effect")
    ax.set_title(
        f"{case['language'].title()}: effect on Gemma score {case['target_label']!r}"
    )
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    return output_path


def plot_pairwise_interactions(
    case: MultilingualCase,
    interactions: np.ndarray,
) -> Path:
    """Plot and save the second-order interaction heatmap."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"multilingual_{case['language']}_pairwise_interactions.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    max_abs = max(abs(interactions.min()), abs(interactions.max()), 1e-6)
    image = ax.imshow(
        interactions,
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(range(len(case["phrase_names"])))
    ax.set_yticks(range(len(case["phrase_names"])))
    ax.set_xticklabels(case["phrase_names"], rotation=25, ha="right")
    ax.set_yticklabels(case["phrase_names"])
    fig.colorbar(image, ax=ax, label="Pairwise interaction")
    ax.set_title(f"{case['language'].title()}: {INTERACTION_INDEX} phrase interactions")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    return output_path


def print_case_summary(
    case: MultilingualCase,
    effects: np.ndarray,
    interactions: np.ndarray,
) -> tuple[str, str, float]:
    """Print and return a short summary for one language."""
    top_positive_idx = int(np.argmax(effects))
    top_negative_idx = int(np.argmin(effects))

    pair_indices = np.triu_indices_from(interactions, k=1)
    pair_values = interactions[pair_indices]

    strongest_position = int(np.argmax(np.abs(pair_values)))
    i = int(pair_indices[0][strongest_position])
    j = int(pair_indices[1][strongest_position])
    strongest_value = float(interactions[i, j])

    print("\nAutomatic summary:")
    print(
        "Top positive phrase: "
        f"{case['phrase_names'][top_positive_idx]} "
        f"({case['phrase_kinds'][top_positive_idx]}), "
        f"effect={effects[top_positive_idx]:.4f}"
    )
    print(
        "Top negative phrase: "
        f"{case['phrase_names'][top_negative_idx]} "
        f"({case['phrase_kinds'][top_negative_idx]}), "
        f"effect={effects[top_negative_idx]:.4f}"
    )
    print(
        "Strongest absolute interaction: "
        f"{case['phrase_names'][i]} + {case['phrase_names'][j]}, "
        f"value={strongest_value:.4f}"
    )

    return case["phrase_names"][i], case["phrase_names"][j], strongest_value


def run_case(
    case: MultilingualCase,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> tuple[str, str, float]:
    """Run one multilingual case and save its figures."""
    text = build_text(case)

    print("\n" + "=" * 80)
    print(f"Language: {case['language']}")
    print(f"Target label: {case['target_label']!r}")
    print("Value function:")
    print("v(S) = log P_Gemma(target_label | selected phrase spans)")

    print("\nFull prompt:")
    print(text)

    print("\nCreating TextImputer...")
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        model_type="causal_lm",
        target_label=case["target_label"],
        prompt_template=PROMPT_TEMPLATE,
        player_level="sentence",
        player_strategy=PhrasePlayerStrategy(case),
        perturbation_type="removal",
        device=DEVICE,
    )

    print("\nPhrase players:")
    players = imputer.player_strategy.get_players()
    for idx, player in enumerate(players):
        print(f"[{idx}] {case['phrase_names'][idx]} ({case['phrase_kinds'][idx]}): {player}")

    n_players = imputer.n_players
    coalitions = all_coalitions(n_players)

    print("\nNumber of players:", n_players)
    print("Number of coalitions:", len(coalitions))

    full_score = imputer.full_prediction()
    print(f"\nFull sentence score for {case['target_label']!r}: {full_score:.4f}")

    print("\nEvaluating all coalitions...")
    scores = imputer.value_function(coalitions)
    effects = estimate_single_phrase_effects(coalitions, scores)

    print(f"\nComputing formal shapiq {INTERACTION_INDEX} interactions...")
    interactions, interaction_values = compute_shapiq_pairwise_interactions(imputer)

    print("\nSingle phrase effects:")
    for name, kind, effect in zip(
        case["phrase_names"], case["phrase_kinds"], effects, strict=False
    ):
        print(f"{name:16s} ({kind:10s}): {effect:.4f}")

    print(f"\nFormal shapiq {INTERACTION_INDEX} pairwise interactions:")
    print(interactions)
    print("\nFormal interaction values object:")
    print(interaction_values.get_n_order(order=INTERACTION_ORDER))

    summary = print_case_summary(case, effects, interactions)

    single_effects_path = plot_single_phrase_effects(case, effects)
    interactions_path = plot_pairwise_interactions(case, interactions)

    print("\nSaved figures:")
    print(single_effects_path)
    print(interactions_path)

    return summary


def main() -> None:
    """Run the multilingual robustness demo."""
    print("Multilingual robustness demo")
    print("Question: Are Shapley interaction explanations stable across translations?")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

    summaries = []
    for case in CASES:
        phrase_i, phrase_j, value = run_case(case, model, tokenizer)
        summaries.append((case["language"], phrase_i, phrase_j, value))

    print("\n" + "=" * 80)
    print("Cross-language comparison:")
    for language, phrase_i, phrase_j, value in summaries:
        print(
            f"{language:10s}: strongest interaction = "
            f"{phrase_i} + {phrase_j}, value={value:.4f}"
        )

    unique_pairs = {(phrase_i, phrase_j) for _, phrase_i, phrase_j, _ in summaries}
    if len(unique_pairs) == 1:
        print("\nConclusion: the strongest interaction is stable across languages.")
    else:
        print("\nConclusion: the strongest interaction changes across languages.")


if __name__ == "__main__":
    main()
