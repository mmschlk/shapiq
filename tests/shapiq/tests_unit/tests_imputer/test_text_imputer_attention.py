from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shapiq.game_theory.exact import ExactComputer
from shapiq.imputer.text_imputer import TextImputer

MODEL_NAME = "distilgpt2"


@pytest.fixture(scope="module")
def causal_lm_components():
    """Load a small causal LM for smoke tests."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    return model, tokenizer, device


def test_causal_lm_removal_strategy_returns_scores(causal_lm_components):
    """Test string-level removal perturbation with a causal LM."""
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
            [True, True, True, True, True, True, True],
            [False, False, False, False, False, False, False],
        ],
        dtype=bool,
    )

    scores = imputer(coalitions)

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_causal_lm_neutral_strategy_returns_scores(causal_lm_components):
    """Test string-level neutral replacement with a causal LM."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="The capital of France is Paris.",
        batch_size=2,
        device=device,
        player_level="word",
        perturbation_type="neutral",
        model_type="causal_lm",
        target_label=" Paris",
        prompt_template="{text}\n\nQuestion: What is the capital of France?\nAnswer:",
    )

    full_coalition = np.ones((1, imputer.n_players), dtype=bool)
    scores = imputer(full_coalition)

    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_causal_lm_pad_strategy_returns_scores(causal_lm_components):
    """Test string-level PAD replacement with a causal LM."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="The capital of France is Paris.",
        batch_size=2,
        device=device,
        player_level="word",
        perturbation_type="pad",
        model_type="causal_lm",
        target_label=" Paris",
        prompt_template="{text}\n\nQuestion: What is the capital of France?\nAnswer:",
    )

    empty_coalition = np.zeros((1, imputer.n_players), dtype=bool)
    scores = imputer(empty_coalition)

    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_attention_mask_strategy_returns_scores(causal_lm_components):
    """Test tensor-level attention masking with sentence players."""
    model, tokenizer, device = causal_lm_components

    text = (
        "Retrieved chunk 1: The capital of France is Paris. "
        "Retrieved chunk 2: The capital of France is Berlin."
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        batch_size=1,
        device=device,
        player_level="sentence",
        perturbation_type="attention_mask",
        model_type="causal_lm",
        attention_prompt_suffix="\n\nQuestion: What is the capital of France?\nAnswer:",
        attention_target_text=" Paris",
        player_separator="\n",
    )

    coalitions = np.array(
        [
            [True, False],
            [False, True],
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )

    scores = imputer(coalitions)

    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))


def test_attention_mask_strategy_with_exact_computer(causal_lm_components):
    """Test that TextImputer with attention_mask works as a shapiq game."""
    model, tokenizer, device = causal_lm_components

    text = (
        "Retrieved chunk 1: The capital of France is Paris. "
        "Retrieved chunk 2: The capital of France is Berlin."
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        batch_size=1,
        device=device,
        player_level="sentence",
        perturbation_type="attention_mask",
        model_type="causal_lm",
        attention_prompt_suffix="\n\nQuestion: What is the capital of France?\nAnswer:",
        attention_target_text=" Paris",
        player_separator="\n",
    )

    exact = ExactComputer(
        n_players=imputer.n_players,
        game=imputer,
    )

    sv = exact(index="SV", order=1)

    assert sv.n_players == 2
    assert np.isfinite(sv.baseline_value)


def test_attention_mask_coalition_to_text_raises(causal_lm_components):
    """Attention masking should not produce a perturbed string."""
    model, tokenizer, device = causal_lm_components

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="The capital of France is Paris.",
        batch_size=1,
        device=device,
        player_level="word",
        perturbation_type="attention_mask",
        model_type="causal_lm",
        attention_prompt_suffix="\n\nQuestion: What is the capital of France?\nAnswer:",
        attention_target_text=" Paris",
        player_separator=" ",
    )

    coalition = np.ones(imputer.n_players, dtype=bool)

    with pytest.raises(TypeError, match="does not produce perturbed text"):
        imputer.coalition_to_text(coalition)


def test_wrong_coalition_width_raises(causal_lm_components):
    """Coalition width must match n_players."""
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

    wrong_coalition = np.array([[True, False]], dtype=bool)

    with pytest.raises(ValueError, match="Expected coalition width"):
        imputer(wrong_coalition)
