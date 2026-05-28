"""Counterfactual explanations.

Given two almost-identical prompts with very
different outputs (e.g. minor negation flip, pronoun swap),
show the interactions responsible forthe divergence.
"""

# ruff: noqa: T201
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shapiq.approximator import PermutationSamplingSII
from shapiq.imputer.text_imputer import TextImputer

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

# ============================================================
# Helper functions for tokenization & interaction formatting.
# ============================================================

def tokenize_words(text: str) -> list[str]:
    """Token rule: 1 Word = 1Player."""
    return re.findall(r"\b[\w']+\b", text.lower())


def top_k_interactions(
    interaction_values: InteractionValues,
    shared_words: list[str],
    k: int = 10,
) -> list[tuple[Any, float]]:
    """Convert top K InteractionValues to readable word tuples."""
    readable = []

    for coalition, value in interaction_values.dict_values.items():
        if len(coalition) == 0:
            continue

        names = tuple(shared_words[i] for i in coalition)
        readable.append((names, float(value)))

    readable.sort(key=lambda x: abs(x[1]), reverse=True)
    return readable[:k]


# ============================================================
# Wrap the TexImputer to operate in a shared coalition space for two sentences (Alignment Problem solved).
# ============================================================

class SharedSpaceTextImputer:
    """shared coalition space -> sentence-local coalition space."""

    def __init__(
        self,
        base_imputer: TextImputer,
        sentence_words: list[str],
        shared_words: list[str],
    ) -> None:
        """Initialize shared-space adapter.

        base_imputer: Original TextImputer for one sentence.
        sentence_words: Tokenized words in this sentence.
        shared_words: Shared vocabulary across both sentences.
        """
        self.base_imputer = base_imputer
        self.sentence_words = sentence_words
        self.shared_words = shared_words
        self.n_players = len(shared_words)

        self.word_to_local_idx = { word: idx for idx, word in enumerate(sentence_words) }

    def _project_shared_to_local(
        self,
        coalition: np.ndarray,
    ) -> np.ndarray:
        """Project a coalition from shared vocabulary space to sentence-local space.

        Parameters:
        ----------
        coalition:
        A single coalition mask defined over the shared vocabulary.
        1 = keep the word
        0 = mask the word

        Returns:
        -------
        np.ndarray: Coalition mask aligned with the current sentence's local word indices.
        """
        local = np.ones(len(self.sentence_words), dtype=int)

        for shared_idx, keep in enumerate(coalition):
            word = self.shared_words[shared_idx]

            if word not in self.word_to_local_idx:
                continue

            local_idx = self.word_to_local_idx[word]
            local[local_idx] = keep

        return local

    def __call__(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate shared-space coalitions using the wrapped sentence-specific imputer.

        Converts coalition masks from the shared vocabulary space into the
        sentence-local feature space, then forwards them to the underlying
        TextImputer for model evaluation.
        """
        coalitions = np.asarray(coalitions)

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        projected = np.array([self._project_shared_to_local(c) for c in coalitions])

        return self.base_imputer(projected)


# ============================================================
# Result Container
# ============================================================

@dataclass
class ContrastiveResult:
    """Container for contrastive explanation results between two sentences.

    Stores the original sentences, prediction scores, shared vocabulary,
    interaction values for each sentence, and readable interaction summaries.
    """
    sentence_a: str
    sentence_b: str
    score_a: float
    score_b: float
    shared_words: list[str]

    iv_a: InteractionValues
    iv_b: InteractionValues
    delta: InteractionValues

    interactions_a: list
    interactions_b: list
    delta_interactions: list


# ============================================================
# Main Explainer
# ============================================================

class ContrastiveTextExplainer:
    """Contrastive explainer for comparing interaction-based explanationsbetween two text inputs."""
    def __init__(
        self,
        model_name: str = "textattack/bert-base-uncased-SST-2",
        class_idx: int = 1,
        batch_size: int = 32,
        n_samples: int = 512,
        max_order: int = 2,
    ) -> None:
        """Initialize the contrastive text explainer.

        Parameters:
        ----------
        model_name: Hugging Face model used for text classification.
        class_idx: Target class index to explain.
        batch_size: Batch size for masked model evaluations.
        n_samples: budgets.
        max_order: Maximum interaction order.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.class_idx = class_idx
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.max_order = max_order

    def _build_shared_vocab(
        self,
        sentence_a: str,
        sentence_b: str,
    )-> tuple[list[str], list[str], list[str]]:
        """Tokenizes both sentences and creates a combined ordered vocabulary containing all unique words from both inputs."""
        words_a = tokenize_words(sentence_a)
        words_b = tokenize_words(sentence_b)

        shared_words = list(dict.fromkeys(words_a + words_b))

        return words_a, words_b, shared_words

    def _make_base_imputer(
        self,
        sentence: str,
    ) -> TextImputer:
        return TextImputer(
            model=self.model,
            tokenizer=self.tokenizer,
            text=sentence,
            class_idx=self.class_idx,
            batch_size=self.batch_size,
            output_type="probability",
        )

    def _explain_sentence(
        self,
        sentence: str,
        sentence_words: list[str],
        shared_words: list[str],
    )-> tuple[InteractionValues, float]:
        """Compute interaction-based explanation for a single sentence.

        Wraps the sentence-specific TextImputer into the shared vocabulary coalition space,
        coalition space, then approximates interaction values using permutation sampling.
        """
        base_imputer = self._make_base_imputer(sentence)

        wrapped = SharedSpaceTextImputer(
            base_imputer=base_imputer,
            sentence_words=sentence_words,
            shared_words=shared_words,
        )

        approximator = PermutationSamplingSII(
            n=wrapped.n_players,
            max_order=self.max_order,
            random_state=42,
        )

        interaction_values = approximator.approximate(
            budget=self.n_samples,
            game=wrapped,
        )

        score = base_imputer.full_prediction()

        return interaction_values, score

    def explain(
        self,
        sentence_a: str,
        sentence_b: str,
    ) -> ContrastiveResult:
        """Generate a contrastive interaction explanation for two sentences."""
        words_a, words_b, shared_words = self._build_shared_vocab(sentence_a, sentence_b)

        iv_a, score_a = self._explain_sentence(
            sentence=sentence_a,
            sentence_words=words_a,
            shared_words=shared_words,
        )

        iv_b, score_b = self._explain_sentence(
            sentence=sentence_b,
            sentence_words=words_b,
            shared_words=shared_words,
        )

        delta = iv_b - iv_a

        readable_a = top_k_interactions(iv_a, shared_words)
        readable_b = top_k_interactions(iv_b, shared_words)
        readable_delta = top_k_interactions(delta, shared_words)

        return ContrastiveResult(
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            score_a=score_a,
            score_b=score_b,
            shared_words=shared_words,
            iv_a=iv_a,
            iv_b=iv_b,
            delta=delta,
            interactions_a=readable_a,
            interactions_b=readable_b,
            delta_interactions=readable_delta,
        )

# ============================================================
# Visualization
# ============================================================

def build_feature_names(
    sentence: str,
    shared_words: list[str],
) -> list[str]:
    """Build display labels for force plot features.

    Keeps the shared coalition dimensionality unchanged while hiding
    words that are not present in the current sentence.
    """
    sentence_words = tokenize_words(sentence)

    return [
        word if word in sentence_words else " "
        for word in shared_words
    ]


def plot_contrastive_force(result: ContrastiveResult) -> None:
    """Show two shapiq force plots with sentence labels."""
    names_a = build_feature_names(
        result.sentence_a,
        result.shared_words,
    )

    names_b = build_feature_names(
        result.sentence_b,
        result.shared_words,
    )

    # sentence A
    result.iv_a.plot_force(
        feature_names=names_a,
        show=False,
    )

    plt.gcf().suptitle(
        f'Sentence A: "{result.sentence_a}"',
        fontsize=16,
        y=0.98,
    )

    # sentence B
    result.iv_b.plot_force(
        feature_names=names_b,
        show=False,
    )

    plt.gcf().suptitle(
        f'Sentence B: "{result.sentence_b}"',
        fontsize=16,
        y=0.98,
    )

    plt.show()


def print_result(result: ContrastiveResult) -> None:
    """Print the contrastive interaction explanation."""
    print("=" * 70)
    print("CONTRASTIVE INTERACTION EXPLANATION")
    print("=" * 70)

    print(f"\nSentence A: {result.sentence_a}")
    print(f"Score A   : {result.score_a:.4f}")

    print(f"\nSentence B: {result.sentence_b}")
    print(f"Score B   : {result.score_b:.4f}")

    print("\nShared vocabulary:")
    print(result.shared_words)

    print("\nTop interactions A:")
    for names, val in result.interactions_a:
        print(f"{names}: {val:.4f}")

    print("\nTop interactions B:")
    for names, val in result.interactions_b:
        print(f"{names}: {val:.4f}")

    print("\nTop contrastive delta:")
    for names, val in result.delta_interactions:
        print(f"{names}: {val:.4f}")


# ============================================================
# Demo execution
# ============================================================

if __name__ == "__main__":
    explainer = ContrastiveTextExplainer(
        model_name="textattack/bert-base-uncased-SST-2",
        n_samples=512,
        max_order=2,
    )

    demo_examples = [
        (
            "Hard vs hardly",
            "He works hard.",
            "He hardly works.",
        ),
        (
            "Length difference",
            "I liked the restaurant.",
            "I liked the restaurant until the food arrived.",
        ),
        (
            "Few vs A few",
            "Few students passed the exam.",
            "A few students passed the exam.",
        ),
    ]

    for title, sentence_a, sentence_b in demo_examples:
        print("\n" + "=" * 80)
        print(f"DEMO: {title}")
        print("=" * 80)

        result = explainer.explain(sentence_a, sentence_b)

        print_result(result)
        plot_contrastive_force(result)