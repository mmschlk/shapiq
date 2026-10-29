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
    LinearGaussianGame,
    LinearGaussianSCM,
    RandomLinearGaussianGame,
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


def _shapley_values(game: LinearGaussianGame) -> np.ndarray:
    values = shapiq.ExactComputer(game)("SV", 1)
    return np.array([values[(player,)] for player in range(game.n_players)])


def _example_scm(noise_cov: np.ndarray | None = None) -> LinearGaussianSCM:
    """X1 and X2 share a hidden cause, and X3 = 2 X1 + X2 + noise."""
    weights = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    return LinearGaussianSCM(
        weights, noise_cov=noise_cov, causal_ordering=[[0, 1], [2]], confounding=[True, False]
    )


def test_linear_gaussian_scm_interventional_mean_by_hand() -> None:
    """do(X1 = 1) leaves X2 at its mean of 0 and moves X3 to 2; the noise covariance is irrelevant."""
    confounded = _example_scm(np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(
        confounded.interventional_mean([True, False, False], np.ones(3)), [1.0, 0.0, 2.0]
    )
    independent = _example_scm()
    for coalition in _all_coalitions(3):
        np.testing.assert_allclose(
            confounded.interventional_mean(coalition, np.ones(3)),
            independent.interventional_mean(coalition, np.ones(3)),
        )


def test_linear_gaussian_scm_samples_follow_its_moments() -> None:
    scm = LinearGaussianSCM.random(5, random_state=0)
    samples = scm.sample(200_000, random_state=0)
    np.testing.assert_allclose(samples.mean(axis=0), scm.mean, atol=0.05)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), scm.cov, rtol=0.05, atol=0.05)
    np.testing.assert_array_equal(scm.sample(3, random_state=1), scm.sample(3, random_state=1))


@pytest.mark.parametrize("confounding_probability", [0.0, 1.0])
def test_random_linear_gaussian_game_causal_equals_do(confounding_probability: float) -> None:
    """On the SCM's own chain graph, conditioning on the covariance reproduces solving the SCM."""
    games = {
        mode: RandomLinearGaussianGame(
            7,
            confounding_probability=confounding_probability,
            mode=mode,
            random_state=0,
            normalize=False,
        )
        for mode in ("causal", "do")
    }
    assert any(games["do"].confounding) == bool(confounding_probability)
    coalitions = _all_coalitions(7)
    np.testing.assert_allclose(
        games["causal"](coalitions), games["do"](coalitions), rtol=1e-10, atol=1e-10
    )


def test_linear_gaussian_game_marginal_shapley_values_are_closed_form() -> None:
    game = RandomLinearGaussianGame(6, mode="marginal", random_state=1)
    expected = game.coefficients * (game.x_explain - game.mean)
    np.testing.assert_allclose(_shapley_values(game), expected, atol=1e-10)


def test_linear_gaussian_game_conditional_values_match_the_precision_form() -> None:
    """Conditioning via the precision matrix, a formula independent of the covariance blocks."""
    game = RandomLinearGaussianGame(6, mode="conditional", random_state=2, normalize=False)
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
def test_linear_gaussian_game_is_efficient(mode: ValueFunctionMode) -> None:
    game = RandomLinearGaussianGame(6, mode=mode, random_state=3)
    total = game.original_model_output - game.empty_value
    assert _shapley_values(game).sum() == pytest.approx(total)


@pytest.mark.parametrize(
    ("confounding", "equivalent_mode"), [([False], "conditional"), ([True], "marginal")]
)
def test_linear_gaussian_game_with_a_misspecified_chain_graph(
    confounding: list[bool], equivalent_mode: ValueFunctionMode
) -> None:
    """One unconfounded group conditions on all intervened features, a confounded one on none."""
    drawn = RandomLinearGaussianGame(6, random_state=4)
    arguments = {"coefficients": drawn.coefficients, "x_explain": drawn.x_explain, "scm": drawn.scm}
    misspecified = LinearGaussianGame(
        **arguments, causal_ordering=[range(6)], confounding=confounding
    )
    equivalent = LinearGaussianGame(**arguments, mode=equivalent_mode)
    do = LinearGaussianGame(**arguments, mode="do")
    coalitions = _all_coalitions(6)
    np.testing.assert_allclose(misspecified(coalitions), equivalent(coalitions))
    assert not np.allclose(misspecified(coalitions), do(coalitions))


def test_random_linear_gaussian_scm_follows_its_causal_ordering() -> None:
    scm = LinearGaussianSCM.random(9, n_groups=4, edge_probability=1.0, random_state=5)
    assert len(scm.causal_ordering) == 4
    position = np.empty(9, dtype=int)
    for index, group in enumerate(scm.causal_ordering):
        position[list(group)] = index
    np.testing.assert_array_equal(scm.weights != 0, position[None, :] < position[:, None])
    for group, confounded in zip(scm.causal_ordering, scm.confounding, strict=True):
        assert not confounded or len(group) > 1


def test_random_linear_gaussian_game_is_seeded() -> None:
    np.testing.assert_array_equal(
        RandomLinearGaussianGame(6, random_state=1).cov,
        RandomLinearGaussianGame(6, random_state=1).cov,
    )
    assert not np.array_equal(
        RandomLinearGaussianGame(6, random_state=1).cov,
        RandomLinearGaussianGame(6, random_state=2).cov,
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
        ({"cov": None}, "either scm or cov"),
        ({"scm": LinearGaussianSCM(np.zeros((2, 2)))}, "either scm or cov"),
        ({"mode": "do"}, "needs an scm"),
        ({"cov": None, "scm": LinearGaussianSCM(np.zeros((3, 3)))}, "scm has 3 features"),
        ({"cov": None, "scm": LinearGaussianSCM(np.zeros((2, 2))), "mean": np.ones(2)}, "mean"),
    ],
)
def test_linear_gaussian_game_rejects_invalid_arguments(kwargs: dict, match: str) -> None:
    valid = {"coefficients": np.array([1.0, 2.0]), "x_explain": np.ones(2), "cov": np.eye(2)}
    with pytest.raises(ValueError, match=match):
        LinearGaussianGame(**{**valid, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weights": np.zeros((2, 3))}, "square"),
        ({"weights": np.array([[0.0, 1.0], [0.0, 0.0]])}, "earlier group"),
        ({"causal_ordering": [[0, 1]]}, "earlier group"),
        ({"noise_cov": np.array([[1.0, 0.5], [0.5, 1.0]])}, "confounded group"),
        (
            {"noise_cov": np.ones((2, 2)), "causal_ordering": [[0, 1]], "confounding": [True]},
            "positive",
        ),
        ({"intercepts": np.ones(3)}, "intercepts has shape"),
    ],
)
def test_linear_gaussian_scm_rejects_invalid_arguments(kwargs: dict, match: str) -> None:
    valid = {"weights": np.array([[0.0, 0.0], [1.0, 0.0]])}
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
        LinearGaussianSCM.random(4, **kwargs)
