"""Word-level interactions in classification (Negation Sample).

For sentiment, NLI, and toxicity tasks, higher-order interactions can capture
phenomena that first-order Shapley values miss, such as negation with
adjectives, subject-verb agreement, multi-word named entities, and sarcasm cues.

negation
subject-verb agreement
named entities
sarcasm
"""

# ruff: noqa: T201
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import shapiq
from shapiq.approximator import PermutationSamplingSII
from shapiq.imputer.text_imputer import TextImputer

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


def print_interactions(
    interaction_values: InteractionValues,
    words: list[str],
) -> None:
    """Print interaction values."""
    print("\n=== Interaction Values ===")

    for coalition, value in zip(
        interaction_values.interaction_lookup,
        interaction_values.values,
        strict=False,
    ):
        if len(coalition) == 1:
            label = words[coalition[0]]

        elif len(coalition) == 2:
            i, j = coalition
            label = f"{words[i]} + {words[j]}"

        else:
            label = str(coalition)

        print(f"{label:<20} {value:.4f}")


def main() -> None:
    """Main function to demonstrate word-level interactions in sentiment analysis."""
    text = "This movie is not bad"

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        class_idx=1,
        batch_size=16,
        output_type="probability",
    )

    print("Original text:", text)
    print("Words:", imputer.words)
    print("Original prediction:", imputer.full_prediction())

    # sanity check
    coalitions = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ]
    )

    print("\n=== Masking Sanity Check ===")
    scores = imputer.value_function(coalitions)

    for coalition, score in zip(coalitions, scores, strict=False):
        masked = imputer.mask_words(coalition)
        print(f"{masked:<40} {score:.4f}")

    approximator = PermutationSamplingSII(
        n=imputer.n_players,
        max_order=2,
        random_state=0,
    )

    interaction_values = approximator.approximate(budget=128, game=imputer)
    print_interactions(interaction_values, imputer.words)
    shapiq.plot.force_plot(
        interaction_values, feature_names=imputer.words, show=True, abbreviate=False
    )


if __name__ == "__main__":
    main()
