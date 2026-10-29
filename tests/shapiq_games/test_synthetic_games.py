"""Tests for the synthetic games and their analytic structure."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

import shapiq
from shapiq.utils import powerset
from shapiq_games.synthetic import (
    SOUM,
    DummyGame,
    LinearGaussianSCM,
    RandomLinearGaussianSCM,
    RandomTableGame,
    UnanimityGame,
)
from shapiq_games.typing import ValueFunctionMode

MODES = get_args(ValueFunctionMode.__value__)


def _all_coalitions(n: int) -> np.ndarray:
    return np.array([[i in s for i in range(n)] for s in powerset(range(n))], dtype=bool)


def test_dummy_game_values_and_counter() -> None:
    game = DummyGame(4, interaction=(1, 2))
    coalitions = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
    np.testing.assert_allclose(game(coalitions), [0.0, 0.25, 1.5, 2.0])
    assert game.access_counter == 4


def test_unanimity_game() -> None:
    game = UnanimityGame(np.array([0, 1, 0, 1]))
    assert game.interaction == (1, 3)
    coalitions = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(game(coalitions), [0.0, 0.0, 1.0, 1.0])


def test_soum_moebius_coefficients_reconstruct_the_game() -> None:
    game = SOUM(6, n_basis_games=12, max_interaction_size=4, random_state=5)
    moebius = game.moebius_coefficients
    coalitions = _all_coalitions(6)
    values = game(coalitions)
    for coalition, value in zip(coalitions, values, strict=True):
        members = set(np.flatnonzero(coalition))
        reconstructed = sum(
            moebius[interaction]
            for interaction in moebius.interaction_lookup
            if set(interaction) <= members
        )
        assert value == pytest.approx(reconstructed)


def test_soum_is_seeded_and_normalizable() -> None:
    np.testing.assert_array_equal(
        SOUM(5, 8, random_state=1).linear_coefficients,
        SOUM(5, 8, random_state=1).linear_coefficients,
    )
    assert not np.array_equal(
        SOUM(5, 8, random_state=1).linear_coefficients,
        SOUM(5, 8, random_state=2).linear_coefficients,
    )
    np.testing.assert_array_equal(SOUM(5, 8).linear_coefficients, SOUM(5, 8).linear_coefficients)
    game = SOUM(5, 8, min_interaction_size=0, random_state=0, normalize=True)
    assert game(game.empty_coalition)[0] == pytest.approx(0.0)


def test_random_table_game_is_a_lookup_table() -> None:
    game = RandomTableGame(4, low=-1.0, high=1.0, random_state=0)
    coalitions = _all_coalitions(4)
    values = game(coalitions)
    assert np.all((values >= -1.0) & (values < 1.0))
    # player i contributes 2**i to the index of a coalition
    indices = coalitions.astype(int) @ (2 ** np.arange(4))
    np.testing.assert_array_equal(values, game.values[indices])
    with pytest.raises(ValueError, match="supports 1 to 20 players"):
        RandomTableGame(21)


def _shapley_values(game: LinearGaussianSCM) -> np.ndarray:
    values = shapiq.ExactComputer(game)("SV", 1)
    return np.array([values[(player,)] for player in range(game.n_players)])


def _interventional_value(game: RandomLinearGaussianSCM, coalition: np.ndarray) -> float:
    """Return f at the mean of X under do(X_S = x*_S), solved from the structural equations."""
    weights = np.where(coalition[:, None], 0.0, game.weights)
    intercepts = np.where(coalition, game.x_explain, game.intercepts)
    means = np.linalg.solve(np.eye(game.n_players) - weights, intercepts)
    return float(game.coefficients @ means)


@pytest.mark.parametrize("confounding_probability", [0.0, 1.0])
def test_random_linear_gaussian_scm_causal_values_are_interventional(
    confounding_probability: float,
) -> None:
    """The chain graph is exact for the drawn model: the causal values are E[f(X) | do(X_S)]."""
    game = RandomLinearGaussianSCM(
        7, confounding_probability=confounding_probability, random_state=0, normalize=False
    )
    assert any(game.confounding) == bool(confounding_probability)
    coalitions = _all_coalitions(7)
    expected = [_interventional_value(game, coalition) for coalition in coalitions]
    np.testing.assert_allclose(game(coalitions), expected, rtol=1e-10, atol=1e-10)


def test_linear_gaussian_scm_marginal_shapley_values_are_closed_form() -> None:
    game = RandomLinearGaussianSCM(6, mode="marginal", random_state=1)
    expected = game.coefficients * (game.x_explain - game.mean)
    np.testing.assert_allclose(_shapley_values(game), expected, atol=1e-10)


def test_linear_gaussian_scm_conditional_values_match_the_precision_form() -> None:
    """Conditioning via the precision matrix, a formula independent of the covariance blocks."""
    game = RandomLinearGaussianSCM(6, mode="conditional", random_state=2, normalize=False)
    precision = np.linalg.inv(game.cov)
    for coalition in _all_coalitions(6)[:-1]:
        present, absent = coalition, ~coalition
        features = game.x_explain.copy()
        residual = game.x_explain[present] - game.mean[present]
        features[absent] = game.mean[absent] - np.linalg.solve(
            precision[np.ix_(absent, absent)], precision[np.ix_(absent, present)] @ residual
        )
        assert game(coalition[None])[0] == pytest.approx(game.coefficients @ features)


@pytest.mark.parametrize("mode", MODES)
def test_linear_gaussian_scm_is_efficient(mode: ValueFunctionMode) -> None:
    game = RandomLinearGaussianSCM(6, mode=mode, random_state=3)
    total = game.original_model_output - game.empty_value
    assert _shapley_values(game).sum() == pytest.approx(total)


@pytest.mark.parametrize(
    ("confounding", "equivalent_mode"), [([False], "conditional"), ([True], "marginal")]
)
def test_linear_gaussian_scm_with_one_group(
    confounding: list[bool], equivalent_mode: ValueFunctionMode
) -> None:
    """One unconfounded group conditions on all intervened features, a confounded one on none."""
    drawn = RandomLinearGaussianSCM(6, random_state=4)
    arguments = {"coefficients": drawn.coefficients, "x_explain": drawn.x_explain, "cov": drawn.cov}
    causal = LinearGaussianSCM(
        **arguments, mean=drawn.mean, causal_ordering=[range(6)], confounding=confounding
    )
    equivalent = LinearGaussianSCM(**arguments, mean=drawn.mean, mode=equivalent_mode)
    coalitions = _all_coalitions(6)
    np.testing.assert_allclose(causal(coalitions), equivalent(coalitions))


def test_random_linear_gaussian_scm_follows_its_causal_ordering() -> None:
    game = RandomLinearGaussianSCM(9, n_groups=4, edge_probability=1.0, random_state=5)
    assert len(game.causal_ordering) == 4
    position = np.empty(9, dtype=int)
    for index, group in enumerate(game.causal_ordering):
        position[group] = index
    np.testing.assert_array_equal(game.weights != 0, position[None, :] < position[:, None])
    for group, confounded in zip(game.causal_ordering, game.confounding, strict=True):
        assert not confounded or len(group) > 1


def test_random_linear_gaussian_scm_is_seeded() -> None:
    np.testing.assert_array_equal(
        RandomLinearGaussianSCM(6, random_state=1).cov,
        RandomLinearGaussianSCM(6, random_state=1).cov,
    )
    assert not np.array_equal(
        RandomLinearGaussianSCM(6, random_state=1).cov,
        RandomLinearGaussianSCM(6, random_state=2).cov,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "interventional"}, "mode must be"),
        ({"x_explain": np.array([1.0])}, "x_explain has shape"),
        ({"cov": np.eye(3)}, "cov has shape"),
        ({"cov": np.array([[1.0, 0.5], [0.0, 1.0]])}, "symmetric"),
        ({"cov": np.ones((2, 2))}, "positive definite"),
        ({"mean": np.array([0.0])}, "mean has shape"),
        ({"causal_ordering": [[0]]}, "must partition"),
        ({"causal_ordering": [[0], [1]], "confounding": [False]}, "confounding has"),
    ],
)
def test_linear_gaussian_scm_rejects_invalid_arguments(kwargs: dict, match: str) -> None:
    valid = {"coefficients": np.array([1.0, 2.0]), "x_explain": np.ones(2), "cov": np.eye(2)}
    with pytest.raises(ValueError, match=match):
        LinearGaussianSCM(**{**valid, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_groups": 0}, "n_groups must be"),
        ({"n_groups": 5}, "n_groups must be"),
        ({"edge_probability": 1.5}, "edge_probability must be"),
        ({"confounding_probability": -0.1}, "confounding_probability must be"),
    ],
)
def test_random_linear_gaussian_scm_rejects_invalid_arguments(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RandomLinearGaussianSCM(4, **kwargs)
