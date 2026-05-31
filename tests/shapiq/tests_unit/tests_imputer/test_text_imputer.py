"""This test module contains test on word-level masking explanations."""

from __future__ import annotations

import numpy as np
import pytest
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shapiq.approximator import PermutationSamplingSII
#from shapiq.imputer.text_imputer import TextImputer
from shapiq.imputer.text_imputer_v4_gemma_supported import TextImputer


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

    #interaction_values = approximator(imputer.value_function)
    interaction_values = approximator(
    budget=100,
    game=imputer.value_function,
)

    #words = imputer.words
    words = imputer.player_strategy.get_players()

    assert words == ["this", "movie", "is", "not", "good"]

    not_idx = words.index("not")
    good_idx = words.index("good")

    pair_value = interaction_values[(not_idx, good_idx)]

    assert np.isfinite(pair_value)

    # Negation should create meaningful interaction
    assert abs(pair_value) > 0.05


# =============================================================================
# Subword Player Strategy Test
# =============================================================================

@pytest.mark.integration
def test_text_imputer_subword_player_distilbert() -> None:
    """Test subword-level players with DistilBERT.
    测试 tokenizer/subword 粒度玩家是否能够正常工作。
    Verify that subword-level players can be created and evaluated.
    """

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text = "unbelievable movie"

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        player_level="subword",
        class_idx=1,
        batch_size=8,
        output_type="logit",
    )

    subwords = imputer.player_strategy.get_players()

    assert imputer.n_players == len(subwords)
    assert imputer.n_players >= 2

    coalition = np.ones(
        imputer.n_players,
        dtype=bool,
    )

    value = imputer.value_function(
        coalition.reshape(1, -1)
    )

    assert np.isfinite(value[0])


# =============================================================================
# Chunk Player Strategy Test
# =============================================================================

@pytest.mark.integration
def test_text_imputer_chunk_player_distilbert() -> None:
    """Test chunk-level players.
    测试 chunk/span 粒度玩家。
    Verify chunk-based player grouping works correctly.
    """

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text = "The big dog runs quickly."

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        player_level="chunk",
        class_idx=1,
        batch_size=8,
        output_type="logit",
    )

    # At least one chunk should exist
    # 至少应识别出一个 chunk
    assert imputer.n_players >= 1

    coalition = np.ones(
        imputer.n_players,
        dtype=bool,
    )

    value = imputer.value_function(
        coalition.reshape(1, -1)
    )

    assert np.isfinite(value[0])


# =============================================================================
# PAD Perturbation Test
# =============================================================================

#@pytest.mark.skip(reason="PadTokenPerturbation has context/_context interface mismatch.")
@pytest.mark.integration
def test_text_imputer_pad_perturbation_distilbert() -> None:
    """Test PAD token replacement.
    测试 PAD 替换策略是否能够正常推理。
    Verify that PAD-token perturbation can be evaluated.
    """

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text = "this movie is very good"

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        perturbation_type="pad",
        class_idx=1,
        batch_size=8,
        output_type="logit",
    )

    coalition = np.zeros(
        imputer.n_players,
        dtype=bool,
    )

    value = imputer.value_function(
        coalition.reshape(1, -1)
    )

    assert np.isfinite(value[0])


# =============================================================================
# Sentence Player Strategy Test
# =============================================================================

@pytest.mark.integration
def test_text_imputer_sentence_player_distilbert() -> None:
    """Test sentence-level players.
    测试 sentence 粒度玩家。
    Verify sentence-based grouping works correctly.
    """

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text = (
        "This movie is great. "
        "The acting is excellent. "
        "I would watch it again."
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        player_level="sentence",
        class_idx=1,
        batch_size=8,
        output_type="logit",
    )

    assert imputer.n_players >= 3

    coalition = np.ones(
        imputer.n_players,
        dtype=bool,
    )

    value = imputer.value_function(
        coalition.reshape(1, -1)
    )

    assert np.isfinite(value[0])


# =============================================================================
# Named Entity Player Strategy Test
# =============================================================================

@pytest.mark.integration
def test_text_imputer_named_entity_player_distilbert() -> None:
    """Test named-entity players.
    测试实体级别玩家。
    Verify named entity grouping works correctly.
    """

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text = (
        "Barack Obama visited New York "
        "and met Microsoft executives."
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        player_level="named_entity",
        class_idx=1,
        batch_size=8,
        output_type="logit",
    )

    assert imputer.n_players >= 1

    coalition = np.ones(
        imputer.n_players,
        dtype=bool,
    )

    value = imputer.value_function(
        coalition.reshape(1, -1)
    )

    assert np.isfinite(value[0])