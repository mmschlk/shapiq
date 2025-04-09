"""This test module contains all tests for the CoalitionSampler class."""

import numpy as np
import pytest

from shapiq.approximator.sampling import CoalitionSampler


@pytest.mark.parametrize("n_players", [5, 10, 50])
@pytest.mark.parametrize("budget", [10, 100])
def test_basic_functionality(n_players, budget):
    """This test checks the basic functionality of the CoalitionSampler class."""
    n = n_players
    expected_budget = min(budget, 2**n)

    # test init and default params
    uniform_sampling_weights = np.ones(n + 1) / (n + 1)  # only empty and full should be complete
    sampler = CoalitionSampler(n, uniform_sampling_weights)
    assert sampler.n == n
    assert np.isclose(sampler._sampling_weights, uniform_sampling_weights).all()
    assert sampler.pairing_trick is False  # default to False
    assert sampler.coalitions_matrix is None
    assert sampler.coalitions_counter is None
    assert sampler.coalitions_probability is None
    assert sampler.n_max_coalitions == 2**n
    assert sampler.n_coalitions == 0
    assert sampler.empty_coalition_index is None

    # test sampling
    sampler.sample(budget)

    # test for correct shape and values
    assert sampler.coalitions_matrix.shape[0] == expected_budget
    assert sampler.coalitions_counter.shape[0] == expected_budget
    assert sampler.coalitions_probability.shape[0] == expected_budget
    assert sampler.n_coalitions == expected_budget
    index_empty = np.where(np.sum(sampler.coalitions_matrix, axis=1) == 0)[0][0]  # all rows zero
    assert sampler.empty_coalition_index == index_empty
    assert len(sampler.coalitions_size) == expected_budget

    # check for correct data types in properties
    assert sampler.is_coalition_size_sampled.dtype == bool
    assert sampler.is_coalition_sampled.dtype == bool
    assert sampler.sampling_adjustment_weights.dtype == float
    assert sampler.sampling_size_probabilities.dtype == float
    assert sampler.coalitions_probability.dtype == float
    assert sampler.coalitions_size_probability.dtype == float
    assert sampler.coalitions_in_size_probability.dtype == float

    # check for error if sampling budget is less than two (empty and full)
    with pytest.raises(ValueError):
        sampler.sample(1)

    # test with pairing
    sampler = CoalitionSampler(n, uniform_sampling_weights, pairing_trick=True)
    assert sampler.pairing_trick is True
    sampler.sample(budget)
    assert sampler.coalitions_matrix.shape[0] == expected_budget
    assert sampler.coalitions_counter.shape[0] == expected_budget
    assert sampler.coalitions_probability.shape[0] == expected_budget
    assert sampler.n_coalitions == expected_budget

    # test for asymmetric sampling weights and pairing trick
    asymmetric_sampling_weights = np.ones(n + 1) / (n + 1)
    asymmetric_sampling_weights[1] = 0.5
    with pytest.warns(UserWarning):
        _ = CoalitionSampler(n, asymmetric_sampling_weights, pairing_trick=True)

    # test for negative sampling weights
    negative_sampling_weights = np.ones(n + 1) / (n + 1)
    negative_sampling_weights[1] = -0.5
    with pytest.raises(ValueError):
        _ = CoalitionSampler(n, negative_sampling_weights)

    # test for mismatch in player number and sampling weights
    with pytest.raises(ValueError):
        _ = CoalitionSampler(n, np.ones(n))

    # test double sampling
    n_first = expected_budget
    n_second = expected_budget - 5
    sampler = CoalitionSampler(n, uniform_sampling_weights, pairing_trick=True)
    sampler.sample(n_first)
    assert sampler.n_coalitions == n_first
    sampler.sample(n_second)
    assert sampler.n_coalitions == n_second


def test_user_warning_stalling():
    """This test checks the warning for sketchy budgets in the CoalitionSampler class."""
    n = 5
    uniform_sampling_weights = np.ones(n + 1) / (n + 1)  # only empty and full

    # test for warning with sketchy budget (stalling)
    with pytest.warns(UserWarning):
        sampler = CoalitionSampler(n, uniform_sampling_weights)
        sampler.sample(2**n - 1)


def test_sampling():
    """This test checks the sampling functionality of the CoalitionSampler class."""
    for random_state in range(30):
        rng = np.random.default_rng(seed=random_state)

        # setup variables for test
        n = rng.integers(low=2, high=12)
        budget = rng.integers(low=1, high=2**12)
        sampling_weights = rng.random(size=n + 1)
        excluded_size = rng.integers(0, high=2, size=n + 1)
        sampling_weights[excluded_size] = 0
        sampling_weights[-2] = 0.5  # ensure one set size remains

        # run the sampler
        sampler = CoalitionSampler(
            n,
            sampling_weights,
            pairing_trick=False,
            random_state=random_state,
        )
        sampler.sample(budget)

        # get params from sampler
        max_samples_in_sampler = min(sampler.n_max_coalitions, budget)
        coalitions_matrix = sampler.coalitions_matrix

        # assert number of unique coalitions
        n_unique_coals = np.unique(coalitions_matrix, axis=0).shape[0]
        assert n_unique_coals == max_samples_in_sampler

        # assert that coalitions counter larger than sampling_budget
        sampled_coalitions_counter = sampler.coalitions_counter
        assert np.sum(sampled_coalitions_counter) >= max_samples_in_sampler

        # assert similar output with scaled sampling weights and similar random_state
        sampling_weights_scaled = sampling_weights * 1.4
        sampler_scaled = CoalitionSampler(
            n,
            sampling_weights_scaled,
            pairing_trick=False,
            random_state=random_state,
        )
        sampler_scaled.sample(budget)
        coalitions_matrix_scaled = sampler_scaled.coalitions_matrix
        assert np.allclose(coalitions_matrix, coalitions_matrix_scaled)  # all equal

        # assert splitting of coalition sizes is correct
        combined_sizes: list = (
            sampler._coalitions_to_sample
            + sampler._coalitions_to_compute
            + sampler._coalitions_to_exclude
        )
        assert len(combined_sizes) == n + 1

        # assert coalitions are excluded from sampling
        for exclude_size in sampler._coalitions_to_exclude:
            assert np.all(np.sum(coalitions_matrix, axis=1) != exclude_size)

        # assert weights are equal to one for fully computed subset sizes
        for compute_size in sampler._coalitions_to_compute:
            assert np.all(
                sampler.coalitions_probability[
                    np.where(np.sum(coalitions_matrix, axis=1) == compute_size)
                ]
                == 1,
            )

        # assert weights are less than one for sampled subsets
        for sample_size in sampler._coalitions_to_sample:
            assert np.all(
                sampler.coalitions_probability[
                    np.where(np.sum(coalitions_matrix, axis=1) == sample_size)
                ]
                < 1,
            )
