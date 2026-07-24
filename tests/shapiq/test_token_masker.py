"""Tests for token masking: positions as players, mask-token baseline."""

from __future__ import annotations

from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    SV,
    BaselineMasker,
    DenseCoalitionArray,
    ExactExplainer,
    MaskedGame,
    ModelMaskedPredictor,
    TokenMasker,
)

N_TOKENS = 4
TOKEN_IDS = [7, 2, 9, 2]
MASK_ID = 0
VALUE_TABLE = np.asarray([0.0, 0.1, -0.5, 0.2, 0.3, -0.1, 0.4, 1.5, 0.6, 2.0])


def all_coalitions():
    rows = jnp.asarray(list(product([False, True], repeat=N_TOKENS)))
    return DenseCoalitionArray(rows)


def test_scalar_mask_tokens_broadcast_and_ids_stay_integers():
    masker = TokenMasker(inputs=np.asarray(TOKEN_IDS), baseline=MASK_ID)
    assert isinstance(masker, BaselineMasker)
    assert masker.n_players == N_TOKENS
    masked = masker(DenseCoalitionArray(jnp.asarray([[True, False, True, False]])))
    assert np.issubdtype(masked.dtype, np.integer)
    assert masked.tolist() == [[7, MASK_ID, 9, MASK_ID]]


def test_per_position_baselines_still_work():
    pads = np.asarray([1, 1, 1, 1])
    masker = TokenMasker(inputs=np.asarray(TOKEN_IDS), baseline=pads)
    masked = masker(DenseCoalitionArray(jnp.asarray([[False, True, False, True]])))
    assert masked.tolist() == [[1, 2, 1, 2]]


def test_backends_agree_on_masked_sequences():
    numpy_masker = TokenMasker(inputs=np.asarray(TOKEN_IDS), baseline=MASK_ID)
    jax_masker = TokenMasker(inputs=jnp.asarray(TOKEN_IDS), baseline=MASK_ID)
    numpy_masked = numpy_masker(all_coalitions())
    jax_masked = jax_masker(all_coalitions())
    assert isinstance(numpy_masked, np.ndarray)
    assert np.array_equal(numpy_masked, np.asarray(jax_masked))


def test_torch_sequences_stay_torch():
    torch = pytest.importorskip("torch")
    masker = TokenMasker(inputs=torch.tensor(TOKEN_IDS), baseline=MASK_ID)
    masked = masker(all_coalitions())
    assert isinstance(masked, torch.Tensor)
    assert masked.dtype == torch.int64
    numpy_masker = TokenMasker(inputs=np.asarray(TOKEN_IDS), baseline=MASK_ID)
    assert np.array_equal(masked.numpy(), np.asarray(numpy_masker(all_coalitions())))


def test_shapley_values_of_an_additive_token_scorer_are_analytic():
    # score = sum of a per-token value: SV of position i is value[id_i] - value[mask]
    def score(token_ids):
        return VALUE_TABLE[np.asarray(token_ids)].sum(axis=-1)

    masker = TokenMasker(inputs=np.asarray(TOKEN_IDS), baseline=MASK_ID)
    game = MaskedGame(masked_predictor=ModelMaskedPredictor(masker=masker, model=score))
    explanation = ExactExplainer(game, SV()).estimate().view
    for position, token in enumerate(TOKEN_IDS):
        expected = VALUE_TABLE[token] - VALUE_TABLE[MASK_ID]
        assert jnp.allclose(explanation((position,)), expected, atol=1e-6)
    assert jnp.allclose(explanation.baseline, N_TOKENS * VALUE_TABLE[MASK_ID], atol=1e-6)
