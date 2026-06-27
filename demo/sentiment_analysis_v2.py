"""Word-level interactions in classification (Multiple examples).

For sentiment, NLI, and toxicity tasks, higher-order interactions can capture
phenomena that first-order Shapley values miss, such as negation with
adjectives, subject-verb agreement, multi-word named entities, and sarcasm cues.

negation
named entities
Contrastive Sentiment
{subject-verb agreement}
{sarcasm}
"""

# ruff: noqa: T201
from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import shapiq
from shapiq.imputer.text_imputer import TextImputer

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


EXAMPLES = [
    {
        "name": "Negation",
        "text": "This movie is not bad",
        "description": ("Negation"),
    },
    {
        "name": "Multi-word Named Entity",
        "text": "I hate fast food.",
        "description": ("Multi-word Named Entity: "),
    },
    {
        "name": "Contrastive Sentiment",
        "text": "Looks good but tastes awful",
        "description": ("Contrastive Sentiment"),
    },
]


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

        print(f"{label:<35} {value:.4f}")


def run_example(
    *,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    title: str,
    description: str,
) -> None:
    """Run one exact interaction demo."""
    print("\n" + "=" * 80)
    print(f"Example: {title}")
    print(description)
    print("=" * 80)

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        class_idx=1,  # positive sentiment
        batch_size=16,
        output_type="probability",
    )

    print("Input text:", text)
    print("Words:", imputer.words)
    print("Original prediction:", imputer.full_prediction())

    computer = shapiq.ExactComputer(
        n_players=imputer.n_players,
        game=imputer,
    )

    interaction_values = computer(
        index="k-SII",
        order=2,
    )

    print_interactions(interaction_values, imputer.words)

    shapiq.plot.force_plot(
        interaction_values, feature_names=imputer.words, show=True, abbreviate=True
    )


def main() -> None:
    """Run multiple word-level interaction demos."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    for example in EXAMPLES:
        run_example(
            model=model,
            tokenizer=tokenizer,
            text=example["text"],
            title=example["name"],
            description=example["description"],
        )


if __name__ == "__main__":
    main()
