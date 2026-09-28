"""Tests for the linear-Gaussian SCM games."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.game_theory.exact import ExactComputer
from shapiq_games.synthetic import ConfoundedChainSCM, LinearGaussianSCM

# Closed-form Shapley values of ConfoundedChainSCM, see the class docstring.
EXPECTED_SHAPLEY_VALUES = {
    "marginal": np.array([1.0, 2.0, 3.0]),
    "conditional": np.array([3.9484, 4.3841, -2.3325]),
    "causal": np.array([4.0, 3.5, -1.5]),
}


def _shapley_values(game: LinearGaussianSCM) -> np.ndarray:
    exact = ExactComputer(n_players=game.n_players, game=game)
    values = exact(index="SV", order=1)
    return np.array([values[(player,)] for player in range(game.n_players)])


@pytest.mark.parametrize("mode", ["marginal", "conditional", "causal"])
def test_confounded_chain_matches_closed_form(mode: str) -> None:
    """The three value functions reproduce their analytically derived attributions."""
    game = ConfoundedChainSCM(mode=mode)
    np.testing.assert_allclose(_shapley_values(game), EXPECTED_SHAPLEY_VALUES[mode], atol=1e-4)


@pytest.mark.parametrize("mode", ["marginal", "conditional", "causal"])
def test_confounded_chain_is_efficient(mode: str) -> None:
    """All three value functions satisfy efficiency at the explanation point."""
    game = ConfoundedChainSCM(mode=mode)
    assert _shapley_values(game).sum() == pytest.approx(6.0)


def test_marginal_and_causal_agree_on_the_sink() -> None:
    """Intervening on a feature with no descendants cannot differ from perturbing it."""
    sink = np.array([[0, 0, 1]]).astype(bool)
    marginal = ConfoundedChainSCM(mode="marginal")(sink)
    causal = ConfoundedChainSCM(mode="causal")(sink)
    np.testing.assert_allclose(marginal, causal)
    np.testing.assert_allclose(marginal, [3.0])


def test_conditional_and_causal_agree_on_the_full_source_group() -> None:
    """Fixing every ancestor leaves nothing for an intervention to cut."""
    sources = np.array([[1, 1, 0]]).astype(bool)
    conditional = ConfoundedChainSCM(mode="conditional")(sources)
    causal = ConfoundedChainSCM(mode="causal")(sources)
    np.testing.assert_allclose(conditional, causal)
    np.testing.assert_allclose(conditional, [12.0])


def test_confounding_flag_changes_the_causal_value() -> None:
    """Declaring the first group unconfounded lets an intervention inform its sibling."""
    coalition = np.array([[1, 0, 0]]).astype(bool)
    confounded = ConfoundedChainSCM(mode="causal")(coalition)
    unconfounded = LinearGaussianSCM(
        coefficients=ConfoundedChainSCM.COEFFICIENTS,
        x_explain=ConfoundedChainSCM.X_EXPLAIN,
        cov=ConfoundedChainSCM.COV,
        causal_ordering=[[0, 1], [2]],
        confounding=[False, False],
        mode="causal",
    )(coalition)
    assert confounded != pytest.approx(unconfounded)


def test_causal_without_ordering_equals_conditional() -> None:
    """One single group means every feature is conditioned on, as in the observational case."""
    coalitions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]]).astype(bool)
    conditional = ConfoundedChainSCM(mode="conditional")(coalitions)
    causal = LinearGaussianSCM(
        coefficients=ConfoundedChainSCM.COEFFICIENTS,
        x_explain=ConfoundedChainSCM.X_EXPLAIN,
        cov=ConfoundedChainSCM.COV,
        mode="causal",
    )(coalitions)
    np.testing.assert_allclose(conditional, causal)


def test_nonzero_mean_shifts_the_empty_coalition() -> None:
    """The normalization value follows the feature mean."""
    game = LinearGaussianSCM(
        coefficients=np.array([1.0, 2.0]),
        x_explain=np.array([1.0, 1.0]),
        cov=np.eye(2),
        mean=np.array([1.0, 1.0]),
        mode="marginal",
        normalize=False,
    )
    empty = np.array([[0, 0]]).astype(bool)
    np.testing.assert_allclose(game(empty), [3.0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "interventional"}, "mode must be"),
        ({"x_explain": np.array([1.0])}, "x_explain has shape"),
        ({"cov": np.eye(3)}, "cov has shape"),
        ({"mean": np.array([0.0])}, "mean has shape"),
        ({"causal_ordering": [[0]]}, "must partition"),
        ({"causal_ordering": [[0], [1]], "confounding": [False]}, "confounding has"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict, match: str) -> None:
    """Malformed inputs fail loudly rather than silently producing a different game."""
    valid = {
        "coefficients": np.array([1.0, 2.0]),
        "x_explain": np.array([1.0, 1.0]),
        "cov": np.eye(2),
    }
    with pytest.raises(ValueError, match=match):
        LinearGaussianSCM(**{**valid, **kwargs})
