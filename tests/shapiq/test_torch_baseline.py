"""Tests for baseline masking of torch models."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

torch = pytest.importorskip("torch")

from shapiq import (  # noqa: E402
    SV,
    BaselineMasker,
    DenseCoalitionArray,
    ExactExplainer,
    MaskedGame,
    ModelMaskedPredictor,
    PermutationSampling,
)
from shapiq.games.torch import to_jax  # noqa: E402

N_PLAYERS = 4
X = torch.tensor([1.0, -2.0, 3.0, 0.5])
BASELINE = torch.tensor([0.5, 0.0, -1.0, 0.5])


def coalitions(rows):
    return DenseCoalitionArray(jnp.asarray(rows, dtype=bool))


def test_masker_replaces_absent_players_with_the_baseline():
    masker = BaselineMasker(inputs=X, baseline=BASELINE)
    assert masker.n_players == N_PLAYERS
    assert masker.target_shape == ()
    masked = masker(coalitions([[True, False, True, False], [False, False, False, False]]))
    expected = torch.tensor([[1.0, 0.0, 3.0, 0.5], [0.5, 0.0, -1.0, 0.5]])
    assert torch.equal(masked, expected)


def test_masker_explains_input_batches():
    batch = torch.stack([X, 2 * X])
    masker = BaselineMasker(inputs=batch, baseline=BASELINE)
    assert masker.target_shape == (2,)
    masked = masker(coalitions([[True, True, True, True]]))
    assert masked.shape == (2, 1, N_PLAYERS)
    assert torch.equal(masked[1, 0], 2 * X)


def test_masker_validates_its_metadata():
    with pytest.raises(ValueError, match="player axis"):
        BaselineMasker(inputs=torch.tensor(1.0), baseline=BASELINE)
    with pytest.raises(ValueError, match=r"baseline must have shape \(4,\)"):
        BaselineMasker(inputs=X, baseline=torch.tensor([0.0, 0.0]))


def linear_vector_game():
    weight = torch.tensor([[1.0, -0.5], [0.0, 2.0], [-1.0, 1.0], [0.5, 0.5]])
    masker = BaselineMasker(inputs=X, baseline=BASELINE)
    predictor = ModelMaskedPredictor(masker=masker, model=lambda inputs: inputs @ weight)
    return MaskedGame(masked_predictor=predictor, link_function=to_jax, value_shape=(2,)), weight


def test_masked_linear_model_has_closed_form_shapley_values():
    game, weight = linear_vector_game()
    explanation = ExactExplainer(game, SV()).explain()
    expected = jnp.asarray(((X - BASELINE)[:, None] * weight).numpy())
    for player in range(N_PLAYERS):
        assert jnp.allclose(explanation((player,)), expected[player], atol=1e-6)
    assert jnp.allclose(explanation.baseline, jnp.asarray((BASELINE @ weight).numpy()), atol=1e-6)


def test_sampled_masked_model_matches_the_exact_explainer():
    game, _ = linear_vector_game()
    exact = ExactExplainer(game, SV()).explain()
    approximator = PermutationSampling(game, SV(), random_state=0)
    estimate = approximator.sample(approximator.min_budget).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(estimate((player,)), exact((player,)), atol=1e-5)


def test_scalar_link_reduces_predictions():
    weight = torch.tensor([[1.0, -0.5], [0.0, 2.0], [-1.0, 1.0], [0.5, 0.5]])
    masker = BaselineMasker(inputs=X, baseline=BASELINE)
    predictor = ModelMaskedPredictor(masker=masker, model=lambda inputs: inputs @ weight)
    game = MaskedGame(
        masked_predictor=predictor,
        link_function=lambda predictions: to_jax(predictions[..., 1]),
    )
    explanation = ExactExplainer(game, SV()).explain()
    expected = jnp.asarray(((X - BASELINE) * weight[:, 1]).numpy())
    for player in range(N_PLAYERS):
        assert jnp.allclose(explanation((player,)), expected[player], atol=1e-6)
