"""This test module contains test on word-level masking explanations."""

from __future__ import annotations

import numpy as np
import pytest
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shapiq.approximator import PermutationSamplingSII
from shapiq.imputer.text_imputer import TextImputer


@pytest.mark.integration
def test_text_imputer_negation_interaction_distilbert() -> None:
    """Test word-level interaction detection with DistilBERT."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text = "this movie is not good"

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        class_idx=1,  # positive class
        batch_size=8,
        output_type="logit",
    )

    approximator = PermutationSamplingSII(
        n=imputer.n_players,
        max_order=2,
        random_state=42,
    )

    interaction_values = approximator(imputer.value_function)

    words = imputer.words

    assert words == ["this", "movie", "is", "not", "good"]

    not_idx = words.index("not")
    good_idx = words.index("good")

    pair_value = interaction_values[(not_idx, good_idx)]

    assert np.isfinite(pair_value)

    # Negation should create meaningful interaction
    assert abs(pair_value) > 0.05
