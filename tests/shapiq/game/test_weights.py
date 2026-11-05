"""Least-squares kernels reproduce their semivalues, and only least-squares indices expose them."""

import jax
import numpy as np
import pytest

from shapiq.game.discretederivative import subset_patterns
from shapiq.game.exact import ExactComputer
from shapiq.game.table import TableGame
from shapiq.game.weights import WeightedBanzhafWeight
from shapiq.index import (
    BanzhafValue,
    LeastSquaresIndex,
    ShapleyInteractionIndex,
    ShapleyValue,
    WeightedBanzhafValue,
)
from shapiq.players import FinitePlayers

N_PLAYERS = 5
TABLE = jax.random.normal(jax.random.key(0), (2**N_PLAYERS,))


def _least_squares_fit(index: LeastSquaresIndex, table: np.ndarray) -> np.ndarray:
    """Fit v(S) ≈ b + sum of S's player values under the kernel; inf entries are constraints."""
    patterns = subset_patterns(N_PLAYERS)
    kernel = np.asarray(index.least_squares_weight(patterns.sum(-1), N_PLAYERS), dtype=float)
    coalitions = np.asarray(patterns)
    design = np.hstack([np.ones((len(coalitions), 1)), coalitions])
    fitted, exact = np.isfinite(kernel), np.isinf(kernel)
    weighted = design[fitted].T * kernel[fitted]
    constraints = design[exact]
    # KKT system of the equality-constrained weighted least squares.
    n_rows = len(constraints)
    lhs = np.block(
        [[weighted @ design[fitted], constraints.T], [constraints, np.zeros((n_rows, n_rows))]]
    )
    rhs = np.concatenate([weighted @ table[fitted], table[exact]])
    return np.linalg.solve(lhs, rhs)[1 : N_PLAYERS + 1]


@pytest.mark.parametrize(
    "index",
    [ShapleyValue(), BanzhafValue(), WeightedBanzhafValue(weights=WeightedBanzhafWeight(0.3))],
)
def test_least_squares_kernel_reproduces_the_semivalue(index):
    exact = ExactComputer.from_game(TableGame(TABLE, FinitePlayers(N_PLAYERS)))(index).values
    fit = _least_squares_fit(index, np.asarray(TABLE, dtype=float))

    assert isinstance(index, LeastSquaresIndex)
    assert np.allclose(fit, exact, atol=1e-5)


def test_only_least_squares_indices_expose_a_kernel():
    # SII carries Shapley weights, but it is not their least-squares fit.
    assert not isinstance(ShapleyInteractionIndex(max_order=2), LeastSquaresIndex)
