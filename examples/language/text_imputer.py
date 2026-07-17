"""Explain a sentiment classifier with the TextImputer.

Run from the project root with:

    uv run python examples/language/text_imputer.py

The first run downloads the Hugging Face model. The word-level player strategy
uses NLTK tokenization; if needed, install the resource once with:

    uv run python -m nltk.downloader punkt_tab

"""

from __future__ import annotations

from itertools import combinations

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as err:
    from shapiq.imputer.text._error import _text_import_error

    raise _text_import_error from err

import nltk


def ensure_nltk_resource(resource: str, download_name: str) -> None:
    try:
        nltk.data.find(resource)

    except LookupError:
        nltk.download(download_name, quiet=True)


ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")

from shapiq.approximator import KernelSHAPIQ
from shapiq.imputer.text.imputer import TextImputer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

TEXT = "The movie is not bad."

EXPLANATION_BUDGET = 64


def select_device() -> str:
    """Select the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """Explain word-level interactions in a sentiment prediction."""
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device).eval()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="encoder_classifier",
        player_level="word",
        perturbation_type="mask",
        class_idx=1,
        output_type="probability",
        device=device,
    )

    words = imputer.player_strategy.get_players()
    normalized_score = float(imputer(imputer.grand_coalition)[0])
    print(f"Input text       : {TEXT}")
    print(f"Word players     : {words}")
    print(f"Normalized score : {normalized_score:.4f}")
    print("Normalized score = full prediction - empty prediction")
    interaction_values = KernelSHAPIQ(
        n=imputer.n_features,
        index="k-SII",
        max_order=2,
        random_state=42,
    ).approximate(
        budget=EXPLANATION_BUDGET,
        game=imputer,
    )
    print("\nFirst-order interaction values")
    print(f"{'idx':>3}  {'word':<18}  {'value':>12}")
    print("-" * 40)
    for idx, word in enumerate(words):
        print(f"{idx:>3}  {word:<18}  {interaction_values[(idx,)]:>+12.4f}")
    print("\nSecond-order interaction values")
    print(f"{'pair':<31}  {'value':>12}")
    print("-" * 47)
    for i, j in combinations(range(imputer.n_features), 2):
        pair_name = f"{words[i]} x {words[j]}"
        print(f"{pair_name:<31}  {interaction_values[(i, j)]:>+12.4f}")
    interaction_values.plot_force(
        feature_names=words,
        show=True,
    )


if __name__ == "__main__":
    main()
