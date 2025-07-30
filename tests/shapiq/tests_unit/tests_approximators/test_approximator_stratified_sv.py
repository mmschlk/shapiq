"""This test module contains all tests regarding the SV stratified sampling approximator."""

from __future__ import annotations

import pytest

from shapiq.approximator.marginals import StratifiedSamplingSV
from shapiq_games.synthetic import DummyGame


@pytest.mark.parametrize(
    ("n", "budget"),
    [(5, 100), (5, 1000)],
)
def test_approximate(n, budget):
    """Tests the approximation of the StratifiedSamplingSV approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = StratifiedSamplingSV(n, random_state=42)

    assert approximator.index == "SV"

    # test for init parameter
    assert approximator.index == "SV"
    assert approximator.n == n
    assert approximator.max_order == 1
    assert approximator.top_order is False

    sv_estimates = approximator.approximate(budget, game)

    # check that the budget is respected
    assert game.access_counter <= budget
    assert sv_estimates.index == "SV"
    assert sv_estimates.max_order == 1
    assert sv_estimates.min_order == 0
    assert sv_estimates.estimation_budget <= budget
    assert sv_estimates.estimated is True  # always estimated

    # check Shapley values for all players that have only marginal contributions of size 0.2
    # their estimates must be exactly 0.2
    assert sv_estimates[(0,)] == pytest.approx(0.2, 0.001)
    assert sv_estimates[(3,)] == pytest.approx(0.2, 0.001)
    assert sv_estimates[(4,)] == pytest.approx(0.2, 0.001)

    # check Shapley values for interaction players
    if budget >= 1000:
        assert sv_estimates[(1,)] == pytest.approx(0.7, 0.2)
        assert sv_estimates[(2,)] == pytest.approx(0.7, 0.2)
