import numpy as np

from shapiq.approximator.sampling import CoalitionSampler


def test_sampling():
    for i in range(100):
        n = np.random.randint(low=1, high=12)
        budget = np.random.randint(low=1, high=2**12)
        random_state = np.random.randint(low=1, high=100)
        # sampling_weights = np.zeros(n + 1)
        sampling_weights = np.random.random(size=n + 1)
        excluded_size = np.random.randint(0, high=2, size=n + 1)
        sampling_weights[excluded_size] = 0
        # Ensure one set size remains
        sampling_weights[-1] = 0.5

        sampler = CoalitionSampler(
            n, sampling_weights, pairing_trick=False, random_state=random_state
        )
        sampler.sample(budget)
        coalitions_matrix = sampler.get_coalitions_matrix()

        # Assert number of unique coalitions
        assert len(np.unique(coalitions_matrix, axis=0)) == min(sampler.n_max_coalitions, budget)

        sampling_weights_scaled = sampling_weights / np.sum(sampling_weights)
        sampler_scaled = CoalitionSampler(
            n, sampling_weights_scaled, pairing_trick=False, random_state=random_state
        )
        sampler_scaled.sample(budget)
        coalitions_matrix_scaled = sampler_scaled.get_coalitions_matrix()

        # Assert similar output with scaled sampling weights and similar random_state
        assert np.sum((coalitions_matrix - coalitions_matrix_scaled) ** 2) < 10e-7

        # Assert that coalitions counter larger than sampling_budget
        sampled_coalitions_counter = sampler.get_coalitions_counter()
        assert np.sum(sampled_coalitions_counter) >= min(sampler.n_max_coalitions, budget)

        sampled_coalitions_weight = sampler.get_coalitions_prob()

        # Assert splitting of coalition sizes is correct
        assert (
            len(
                sampler.coalitions_to_sample
                + sampler.coalitions_to_compute
                + sampler.coalitions_to_exclude
            )
            == n + 1
        )

        for compute_size in sampler.coalitions_to_compute:
            # Assert weights are equal to one for fully computed subset sizes
            assert (
                sampled_coalitions_weight[
                    np.where(np.sum(coalitions_matrix, axis=1) == compute_size)[0]
                ]
                == 1
            ).all()
        for sample_size in sampler.coalitions_to_sample:
            # Assert weights are less than one for sampled subsets
            assert (
                sampled_coalitions_weight[
                    np.where(np.sum(coalitions_matrix, axis=1) == sample_size)[0]
                ]
                < 1
            ).all()
        for exclude_size in sampler.coalitions_to_exclude:
            # Assert coalitions are excluded from sampling
            assert (np.sum(coalitions_matrix, axis=1) != exclude_size).all()
