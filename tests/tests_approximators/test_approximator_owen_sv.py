"""This test module contains all tests regarding the SV stratified sampling approximator."""

import numpy as np
import pytest

from approximator.marginals import OwenSamplingSV
from shapiq.games import DummyGame


@pytest.mark.parametrize(
    "n, m, expected",
    [
        (5, 1, [0.5]),
        (5, 2, [0.0, 1.0]),
        (5, 3, [0.0, 0.5, 1.0]),
        (5, 11, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    ],
)
def test_anchorpoints(n, m, expected):
    """Tests the generation of anchor points in the OwenSamplingSV approximator."""

    approximator = OwenSamplingSV(n, m, random_state=42)
    output = approximator.get_anchor_points(m)
    expected = np.array(expected)
    assert np.allclose(output, expected)


@pytest.mark.parametrize(
    "n, M, budget, batch_size",
    [
        (5, 3, 102, 1),
        (5, 3, 102, 2),
        (5, 3, 102, 5),
        (5, 3, 100, 8),
        (5, 3, 1000, 10),
        (5, 1, 1000, 10),
        (5, 2, 1000, 10),
        (5, 10, 1000, 10),
    ],
)
def test_approximate(n, M, budget, batch_size):
    """Tests the approximation of the OwenSamplingSV approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = OwenSamplingSV(n, M, random_state=42)

    assert approximator.index == "SV"

    # test for init parameter
    assert approximator.index == "SV"
    assert approximator.n == n
    assert approximator.iteration_cost == 2 * n * M
    assert approximator.max_order == 1
    assert approximator.top_order is False

    sv_estimates = approximator.approximate(budget, game, batch_size=batch_size)

    # check that the budget is respected
    assert game.access_counter <= budget

    # check Shapley values for all players that have only marginal contributions of size 0.2
    # their estimates must be exactly 0.2
    assert sv_estimates[(0,)] == pytest.approx(0.2, 0.001)
    assert sv_estimates[(3,)] == pytest.approx(0.2, 0.001)
    assert sv_estimates[(4,)] == pytest.approx(0.2, 0.001)

    # check Shapley values for interaction players
    if budget >= 1000:
        assert sv_estimates[(1,)] == pytest.approx(0.7, 0.1)
        assert sv_estimates[(2,)] == pytest.approx(0.7, 0.1)
