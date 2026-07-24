"""Tests for backend-general masking: one masker, any Array API backend."""

from __future__ import annotations

import subprocess
import sys
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
    SuperpixelMasker,
    grid_labels,
)

N_PLAYERS = 4
INPUTS = [1.0, -2.0, 0.5, 3.0]
BASELINE = [0.0, 1.0, 0.0, -1.0]


def all_coalitions():
    rows = jnp.asarray(list(product([False, True], repeat=N_PLAYERS)))
    return DenseCoalitionArray(rows)


def linear_game(masker):
    def model(masked):
        return np.asarray(masked) @ np.arange(1.0, N_PLAYERS + 1)

    return MaskedGame(masked_predictor=ModelMaskedPredictor(masker=masker, model=model))


def test_numpy_and_jax_maskers_agree():
    numpy_masker = BaselineMasker(inputs=np.asarray(INPUTS), baseline=np.asarray(BASELINE))
    jax_masker = BaselineMasker(inputs=jnp.asarray(INPUTS), baseline=jnp.asarray(BASELINE))
    assert numpy_masker.n_players == jax_masker.n_players == N_PLAYERS
    numpy_masked = numpy_masker(all_coalitions())
    assert isinstance(numpy_masked, np.ndarray)
    jax_masked = jax_masker(all_coalitions())
    assert isinstance(jax_masked, jnp.ndarray)
    assert jnp.allclose(jnp.asarray(numpy_masked), jax_masked)
    assert jnp.allclose(linear_game(numpy_masker)(all_coalitions()),
                        linear_game(jax_masker)(all_coalitions()))


def test_torch_masker_matches_numpy():
    torch = pytest.importorskip("torch")
    torch_masker = BaselineMasker(
        inputs=torch.tensor(INPUTS), baseline=torch.tensor(BASELINE)
    )
    masked = torch_masker(all_coalitions())
    assert isinstance(masked, torch.Tensor)
    numpy_masker = BaselineMasker(inputs=np.asarray(INPUTS), baseline=np.asarray(BASELINE))
    assert np.allclose(masked.numpy(), np.asarray(numpy_masker(all_coalitions())))


def test_mixed_backends_teach():
    with pytest.raises(ValueError, match="one backend"):
        BaselineMasker(inputs=np.asarray(INPUTS), baseline=jnp.asarray(BASELINE))


def test_sklearn_models_explain_without_ceremony():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415 - requires sklearn

    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, N_PLAYERS))
    coefficients = np.asarray([1.0, -2.0, 0.5, 3.0])
    model = LinearRegression().fit(features, features @ coefficients + 4.0)
    background = features.mean(axis=0)
    masker = BaselineMasker(inputs=features[0], baseline=background)
    game = MaskedGame(masked_predictor=ModelMaskedPredictor(masker=masker, model=model.predict))
    explanation = ExactExplainer(game, SV()).estimate().view
    # a linear model's Shapley values are w_i * (x_i - baseline_i)
    for player in range(N_PLAYERS):
        expected = coefficients[player] * (features[0, player] - background[player])
        assert jnp.allclose(explanation((player,)), expected, atol=1e-5)


def test_superpixel_masking_is_backend_general():
    height = width = 4
    labels = grid_labels(height, width, grid=(2, 2))
    assert isinstance(labels, np.ndarray)
    rng = np.random.default_rng(1)
    image = rng.normal(size=(3, height, width))
    numpy_masker = SuperpixelMasker(inputs=image, baseline=0.0, labels=labels)
    jax_masker = SuperpixelMasker(inputs=jnp.asarray(image), baseline=0.0, labels=labels)
    assert numpy_masker.n_players == jax_masker.n_players == 4
    coalitions = DenseCoalitionArray(jnp.asarray([[True, False, False, True]]))
    numpy_masked = numpy_masker(coalitions)
    assert isinstance(numpy_masked, np.ndarray)
    assert jnp.allclose(jnp.asarray(numpy_masked), jax_masker(coalitions))
    # absent superpixels are zeroed, present ones keep their pixels
    masked_image = numpy_masked[0]
    assert np.allclose(masked_image[:, :2, :2], image[:, :2, :2])  # superpixel 0 present
    assert np.allclose(masked_image[:, :2, 2:], 0.0)  # superpixel 1 absent


def test_numpy_masking_never_imports_torch():
    probe = """
import sys
import numpy as np
import jax.numpy as jnp
from shapiq import BaselineMasker, DenseCoalitionArray

masker = BaselineMasker(inputs=np.ones(3), baseline=np.zeros(3))
masker(DenseCoalitionArray(jnp.asarray([[True, False, True]])))
assert "torch" not in sys.modules, "masking numpy inputs must not import torch"
print("lazy")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "lazy" in result.stdout
