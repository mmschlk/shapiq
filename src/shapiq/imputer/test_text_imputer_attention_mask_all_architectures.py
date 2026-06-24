"""All-architecture integration tests for TextImputer attention-mask perturbation.

This file tests attention_mask + ExactComputer for:

Architectures:
1. encoder_classifier
2. causal_lm
3. seq2seq

Player levels:
1. word
2. sentence
3. named_entity
"""

from __future__ import annotations

import numpy as np
import torch
from shapiq import ExactComputer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shapiq.imputer.text_imputer import TextImputer


# =============================================================================
# Shared utilities
# =============================================================================


def _build_basic_coalitions(n_players: int) -> np.ndarray:
    """Build a small sanity-check coalition matrix."""
    full = np.ones(n_players, dtype=bool)
    empty = np.zeros(n_players, dtype=bool)

    if n_players >= 2:
        first_only = np.zeros(n_players, dtype=bool)
        first_only[0] = True

        last_only = np.zeros(n_players, dtype=bool)
        last_only[-1] = True

        return np.array(
            [
                full,
                empty,
                first_only,
                last_only,
            ],
            dtype=bool,
        )

    return np.array([full, empty], dtype=bool)


def _sanity_check_value_function(imputer: TextImputer) -> None:
    """Check that value_function returns scalar scores and full_prediction aligns."""
    coalitions = _build_basic_coalitions(imputer.n_players)
    scores = imputer.value_function(coalitions)

    print("players:")
    for idx, player in enumerate(imputer.player_strategy.get_players()):
        print(f"  {idx}: {player}")

    print("n_players:", imputer.n_players)
    print("sanity coalitions shape:", coalitions.shape)
    print("sanity scores:", scores)
    print("sanity scores shape:", scores.shape)
    print("full_prediction:", imputer.full_prediction())
    print("full == scores[0]:", np.isclose(imputer.full_prediction(), scores[0]))

    assert scores.shape == (coalitions.shape[0],)
    assert np.isclose(imputer.full_prediction(), scores[0])


def _run_exact_computer(imputer: TextImputer):
    """Run ExactComputer and return Shapley values."""
    exact = ExactComputer(
        n_players=imputer.n_players,
        game=imputer.value_function,
    )
    return exact(index="SV")


def _print_exact_summary(title: str, imputer: TextImputer, sv) -> None:
    """Print useful ExactComputer summary."""
    full_prediction = imputer.full_prediction()
    baseline_value = sv.baseline_value

    print(f"\n========== {title} ==========")
    print("full_prediction:", full_prediction)
    print("baseline_value:", baseline_value)
    print("full - baseline:", full_prediction - baseline_value)
    print(sv)


def _run_case(title: str, imputer: TextImputer) -> None:
    """Run sanity checks and ExactComputer for one imputer."""
    print(f"\n\n################################################################################")
    print(f"# {title}")
    print("################################################################################")

    _sanity_check_value_function(imputer)
    sv = _run_exact_computer(imputer)
    _print_exact_summary(title=title, imputer=imputer, sv=sv)


# =============================================================================
# Model builders
# =============================================================================


def build_encoder_classifier():
    """Build a small encoder classifier."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


def build_causal_lm():
    """Build a small causal LM."""
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def build_seq2seq():
    """Build a small seq2seq model."""
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


# =============================================================================
# Text examples
# =============================================================================


TEXT_BY_PLAYER_LEVEL = {
    "word": "This movie is good.",
    "sentence": "The movie started slowly. The ending was surprisingly good.",
    "named_entity": "Barack Obama visited Paris and praised Microsoft.",
}


# =============================================================================
# Encoder classifier tests
# =============================================================================


def test_encoder_classifier_word() -> None:
    model, tokenizer = build_encoder_classifier()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["word"],
        player_level="word",
        perturbation_type="attention_mask",
        model_type="encoder_classifier",
        class_idx=1,
        output_type="probability",
        device="cpu",
        batch_size=16,
    )
    _run_case("encoder_classifier / word / attention_mask", imputer)


def test_encoder_classifier_sentence() -> None:
    model, tokenizer = build_encoder_classifier()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["sentence"],
        player_level="sentence",
        perturbation_type="attention_mask",
        model_type="encoder_classifier",
        class_idx=1,
        output_type="probability",
        device="cpu",
        batch_size=16,
    )
    _run_case("encoder_classifier / sentence / attention_mask", imputer)


def test_encoder_classifier_named_entity() -> None:
    model, tokenizer = build_encoder_classifier()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["named_entity"],
        player_level="named_entity",
        perturbation_type="attention_mask",
        model_type="encoder_classifier",
        class_idx=1,
        output_type="probability",
        device="cpu",
        batch_size=16,
    )
    _run_case("encoder_classifier / named_entity / attention_mask", imputer)


# =============================================================================
# Causal LM tests
# =============================================================================


def test_causal_lm_word() -> None:
    model, tokenizer = build_causal_lm()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["word"],
        player_level="word",
        perturbation_type="attention_mask",
        model_type="causal_lm",
        target_label="good",
        prompt_template="Review: {text}\n\nSentiment:",
        device="cpu",
        batch_size=16,
    )
    _run_case("causal_lm / word / attention_mask", imputer)


def test_causal_lm_sentence() -> None:
    model, tokenizer = build_causal_lm()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["sentence"],
        player_level="sentence",
        perturbation_type="attention_mask",
        model_type="causal_lm",
        target_label="good",
        prompt_template="Review: {text}\n\nSentiment:",
        device="cpu",
        batch_size=16,
    )
    _run_case("causal_lm / sentence / attention_mask", imputer)


def test_causal_lm_named_entity() -> None:
    model, tokenizer = build_causal_lm()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["named_entity"],
        player_level="named_entity",
        perturbation_type="attention_mask",
        model_type="causal_lm",
        target_label="good",
        prompt_template="Review: {text}\n\nSentiment:",
        device="cpu",
        batch_size=16,
    )
    _run_case("causal_lm / named_entity / attention_mask", imputer)


# =============================================================================
# Seq2Seq tests
# =============================================================================


def test_seq2seq_word() -> None:
    model, tokenizer = build_seq2seq()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["word"],
        player_level="word",
        perturbation_type="attention_mask",
        model_type="seq2seq",
        target_label="positive",
        prompt_template="Review: {text}\n\nSentiment:",
        device="cpu",
        batch_size=16,
    )
    _run_case("seq2seq / word / attention_mask", imputer)


def test_seq2seq_sentence() -> None:
    model, tokenizer = build_seq2seq()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["sentence"],
        player_level="sentence",
        perturbation_type="attention_mask",
        model_type="seq2seq",
        target_label="positive",
        prompt_template="Review: {text}\n\nSentiment:",
        device="cpu",
        batch_size=16,
    )
    _run_case("seq2seq / sentence / attention_mask", imputer)


def test_seq2seq_named_entity() -> None:
    model, tokenizer = build_seq2seq()
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT_BY_PLAYER_LEVEL["named_entity"],
        player_level="named_entity",
        perturbation_type="attention_mask",
        model_type="seq2seq",
        target_label="positive",
        prompt_template="Review: {text}\n\nSentiment:",
        device="cpu",
        batch_size=16,
    )
    _run_case("seq2seq / named_entity / attention_mask", imputer)


# =============================================================================
# Main runner
# =============================================================================


def run_all_tests() -> None:
    """Run all attention-mask architecture/player-level integration tests."""
    torch.set_grad_enabled(False)

    test_encoder_classifier_word()
    test_encoder_classifier_sentence()
    test_encoder_classifier_named_entity()

    test_causal_lm_word()
    test_causal_lm_sentence()
    test_causal_lm_named_entity()

    test_seq2seq_word()
    test_seq2seq_sentence()
    test_seq2seq_named_entity()


if __name__ == "__main__":
    run_all_tests()
