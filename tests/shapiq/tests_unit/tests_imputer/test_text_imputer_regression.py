from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shapiq.imputer.text_imputer import TextImputer

CAUSAL_MODEL_NAME = "distilgpt2"
ENCODER_MODEL_NAME = "lvwerra/distilbert-imdb"


@pytest.fixture(scope="module")
def causal_lm_components():
    """Load a small causal LM for TextImputer smoke tests."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(CAUSAL_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(CAUSAL_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    return model, tokenizer, device


@pytest.fixture(scope="module")
def encoder_classifier_components():
    """Load a small encoder classifier for TextImputer smoke tests."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(ENCODER_MODEL_NAME)

    model.to(device)
    model.eval()

    return model, tokenizer, device


def test_word_level_removal_causal_lm_returns_scores(causal_lm_components):
    """Word-level players should still work with removal perturbation."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="The capital of France is Paris.",
        batch_size=2,
        device=device,
        player_level="word",
        perturbation_type="removal",
        model_type="causal_lm",
        target_label=" Paris",
        prompt_template="{text}\n\nQuestion: What is the capital of France?\nAnswer:",
    )

    coalitions = np.array(
        [
            np.ones(imputer.n_players, dtype=bool),
            np.zeros(imputer.n_players, dtype=bool),
        ],
        dtype=bool,
    )

    scores = imputer(coalitions)

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_sentence_level_neutral_causal_lm_returns_scores(causal_lm_components):
    """Sentence-level players should still work with neutral perturbation."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=(
            "The capital of France is Paris. "
            "The capital of Germany is Berlin."
        ),
        batch_size=2,
        device=device,
        player_level="sentence",
        perturbation_type="neutral",
        model_type="causal_lm",
        target_label=" Paris",
        prompt_template="{text}\n\nQuestion: What is the capital of France?\nAnswer:",
    )

    coalitions = np.array(
        [
            np.ones(imputer.n_players, dtype=bool),
            np.zeros(imputer.n_players, dtype=bool),
        ],
        dtype=bool,
    )

    scores = imputer(coalitions)

    assert imputer.n_players == 2
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_chunk_level_removal_causal_lm_returns_scores(causal_lm_components):
    """Chunk-level players should still work with removal perturbation."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="The small cat quickly sleeps.",
        batch_size=2,
        device=device,
        player_level="chunk",
        perturbation_type="removal",
        model_type="causal_lm",
        target_label=" sleeps",
        prompt_template="{text}\n\nWhat happens?\nAnswer:",
    )

    full_coalition = np.ones((1, imputer.n_players), dtype=bool)
    scores = imputer(full_coalition)

    assert imputer.n_players > 0
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_subword_level_removal_causal_lm_returns_scores(causal_lm_components):
    """Subword-level players should still work with removal perturbation."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="Paris is beautiful.",
        batch_size=2,
        device=device,
        player_level="subword",
        perturbation_type="removal",
        model_type="causal_lm",
        target_label=" beautiful",
        prompt_template="{text}\n\nDescription:",
    )

    full_coalition = np.ones((1, imputer.n_players), dtype=bool)
    scores = imputer(full_coalition)

    assert imputer.n_players > 0
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_full_prediction_causal_lm_returns_float(causal_lm_components):
    """full_prediction should still return a scalar float."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="The capital of France is Paris.",
        batch_size=1,
        device=device,
        player_level="word",
        perturbation_type="removal",
        model_type="causal_lm",
        target_label=" Paris",
        prompt_template="{text}\n\nQuestion: What is the capital of France?\nAnswer:",
    )

    score = imputer.full_prediction()

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_encoder_classifier_mask_strategy_returns_scores(encoder_classifier_components):
    """Encoder classifier path should still work with MASK replacement."""
    model, tokenizer, device = encoder_classifier_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="The movie was surprisingly good.",
        batch_size=2,
        device=device,
        player_level="word",
        perturbation_type="mask",
        model_type="encoder_classifier",
        class_idx=1,
        output_type="logit",
    )

    coalitions = np.array(
        [
            np.ones(imputer.n_players, dtype=bool),
            np.zeros(imputer.n_players, dtype=bool),
        ],
        dtype=bool,
    )

    scores = imputer(coalitions)

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
