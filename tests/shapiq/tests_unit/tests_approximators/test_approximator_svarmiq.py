"""This test module contains all tests regarding the svarmiq approximator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shapiq.approximator.montecarlo import SVARMIQ
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame


@pytest.mark.parametrize(
    ("n", "max_order", "index", "top_order"),
    [
        (7, 2, "SII", False),
        (7, 2, "SII", True),
        (7, 2, "k-SII", True),
        (7, 2, "k-SII", False),
        (7, 2, "STII", False),
        (7, 2, "STII", True),
        (7, 2, "FSII", True),
    ],
)
def test_initialization(n, max_order, index, top_order):
    """Tests the initialization of the SVARM-IQ approximator."""
    approximator = SVARMIQ(n, max_order, index=index, top_order=top_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 0)
    assert approximator.iteration_cost == 1
    assert approximator.index == index


@pytest.mark.parametrize(
    ("n", "max_order", "top_order", "budget"),
    [(7, 2, False, 100), (7, 2, True, 100), (7, 2, False, 300)],
)
def test_approximate_sii(n, max_order, top_order, budget):
    """Tests the approximation of the SVARM-IQ SII approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = SVARMIQ(n, max_order, index="SII", top_order=top_order, random_state=42)
    estimates = approximator.approximate(budget, game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 0)

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # for order 2 (max_order) the interaction between player 1 and 2 is the most important (1.0)
    assert estimates[interaction] == pytest.approx(1.0, 0.01)

    if not top_order:
        # for order 1 (min_order) the interaction between  1 and 2 is the most important (0.6429)
        assert estimates[(1,)] == pytest.approx(0.6429, 0.05)
        assert estimates[(2,)] == pytest.approx(0.6429, 0.05)


@pytest.mark.parametrize(("n", "budget"), [(7, 2), (7, 50), (7, 128), (130, 160)])
def test_stratum_adjustments(n, budget):
    """Repeated draws count within strata; enumerated coalitions need no adjustment."""
    approximator = SVARMIQ(n, random_state=42)
    sampler = approximator._sampler
    sampler.sample(budget)
    for interaction in [(), (0,), (0, 2)]:
        observed = approximator._log_svarmiq_routine(interaction)
        expected = []
        for row, count, sampled in zip(
            sampler.coalitions_matrix,
            sampler.coalitions_counter,
            sampler.is_coalition_sampled,
            strict=True,
        ):
            adjustment = math.log(count)
            if sampled:
                size = int(row.sum())
                same_stratum = (
                    np.all(
                        sampler.coalitions_matrix[:, interaction] == row[list(interaction)], axis=1
                    )
                    & (sampler.coalitions_size == size)
                    & sampler.is_coalition_sampled
                )
                adjustment += math.log(
                    math.comb(n - len(interaction), size - int(row[list(interaction)].sum()))
                )
                adjustment -= math.log(sampler.coalitions_counter[same_stratum].sum())
            expected.append(adjustment)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-13)


@pytest.mark.parametrize("index", ["SV", "SII", "k-SII", "STII"])
@pytest.mark.parametrize("pairing", [False, True])
@pytest.mark.parametrize("mutate_coalitions", [False, True])
def test_cached_sampler_matches_uncached_calls(index, pairing, mutate_coalitions):
    """Caching preserves exact results, queried coalitions and RNG across repeated sampling."""
    import copy

    import numpy as np

    class UncachedSVARMIQ(SVARMIQ):
        def _log_svarmiq_routine(self, interaction, **_cached):
            # Exercise the previous per-interaction property-access path.
            return super()._log_svarmiq_routine(interaction)

    options = {
        "n": 7,
        "max_order": 1 if index == "SV" else 2,
        "index": index,
        "pairing_trick": pairing,
        "random_state": 42,
    }
    cached, uncached = SVARMIQ(**options), UncachedSVARMIQ(**options)
    for budget in (40, 128, 63):
        queries = [[], []]

        def game(slot, queries=queries):
            def evaluate(coalitions):
                queries[slot].append(coalitions.copy())
                values = (
                    3.0
                    + coalitions @ np.arange(1.0, 8.0)
                    + 2.5 * np.prod(coalitions[:, [0, 2, 5]], axis=1)
                )
                if mutate_coalitions:
                    coalitions[:, 0] = False
                return values

            return evaluate

        left = cached.approximate(budget, game(0))
        right = uncached.approximate(budget, game(1))
        np.testing.assert_array_equal(left.values, right.values)
        assert left.interaction_lookup == right.interaction_lookup
        assert left.baseline_value == right.baseline_value
        assert left.estimation_budget == right.estimation_budget
        for a, b in zip(queries[0], queries[1], strict=True):
            np.testing.assert_array_equal(a, b)
        for field in ("coalitions_matrix", "coalitions_counter", "is_coalition_sampled"):
            np.testing.assert_array_equal(
                getattr(cached._sampler, field), getattr(uncached._sampler, field)
            )
        assert (
            copy.deepcopy(cached._sampler._rng.bit_generator.state)
            == uncached._sampler._rng.bit_generator.state
        )


def test_svarmiq_does_not_read_sampling_properties_per_interaction(monkeypatch):
    """Property accesses stay bounded when the number of output interactions grows."""
    from collections import Counter

    import numpy as np

    from shapiq.approximator.sampling import CoalitionSampler

    calls = Counter()
    for field in (
        "coalitions_matrix",
        "coalitions_size",
        "coalitions_counter",
        "is_coalition_sampled",
    ):
        getter = getattr(CoalitionSampler, field).fget

        def counted(self, getter=getter, field=field):
            calls[field] += 1
            return getter(self)

        monkeypatch.setattr(CoalitionSampler, field, property(counted))
    counts = []
    for n in (4, 8):
        estimator = SVARMIQ(n, 2, random_state=7)
        estimator._sampler.sample(12)
        matrix = estimator._sampler.coalitions_matrix
        before = matrix.copy()
        calls.clear()
        estimator.monte_carlo_routine(matrix.sum(axis=1), matrix, "SII")
        counts.append(dict(calls))
        np.testing.assert_array_equal(matrix, before)
    assert counts[0] == counts[1]
    assert counts[0]["coalitions_counter"] == 1
    assert counts[0]["is_coalition_sampled"] == 1
