"""This test module contains all tests regarding the SV stratified sampling approximator."""

import numpy as np
import pytest

from shapiq.approximator.marginals import StratifiedSamplingSV
from shapiq.games import DummyGame


@pytest.mark.parametrize(
    "n, budget, batch_size",
    [(5, 102, 1), (5, 102, 2), (5, 102, 5), (5, 100, 8), (5, 1000, 10)],
)
def test_approximate(n, budget, batch_size):
    """Tests the approximation of the StratifiedSamplingSV approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = StratifiedSamplingSV(n, random_state=42)

    assert approximator.index == "SV"

    # test for init parameter
    assert approximator.index == "SV"
    assert approximator.n == n
    assert approximator.iteration_cost == 2 * n * n
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
        assert sv_estimates[(1,)] == pytest.approx(0.7, 0.2)
        assert sv_estimates[(2,)] == pytest.approx(0.7, 0.2)
