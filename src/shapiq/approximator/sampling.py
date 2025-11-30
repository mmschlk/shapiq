"""This module contains stochastic sampling procedures for coalitions of players."""

import numpy as np
import math
from scipy.special import comb as binom
from typing import Sequence, Tuple, TypeVar

class CoalitionSampler:
    '''
    Samples coalitions without replacement according to given sampling weights per coalition size.
    The sampling procedure has two main steps:
    1. Given a budget, compute sampling probabilities per coalition size via closed-form inversion of the expected sample count function.
    2. Sample coalitions of each size according to these probabilities.

    Args:
        n_players (int): Number of players in the game.

        sampling_weights (np.ndarray): Array of sampling weights per coalition size (length n_players-1).

        pairing_trick (bool, optional): Whether to use the pairing trick to reduce computation. Defaults to True.

        random_state (int | None, optional): Random seed for reproducibility

        sample_with_replacement (bool, optional): Whether to sample coalitions with replacement when the number of combinations is too large. Defaults to False.

    All variables are stored in the sampler, no objects are returned. The following variables
    are computed:
        - ``_sampled_coalitions_matrix``: A binary matrix that consists of one row for each sampled
            coalition. Each row is a binary vector that indicates the players in the coalition.
            The matrix is of shape ``(n_coalitions, n_players)``.
        - ``_sampled_coalitions_counter``: An array with the number of occurrences of the coalitions
            in the sampling process. The array is of shape ``(n_coalitions,)``.
        - ``_sampled_coalitions_probability``: An array with the coalition probabilities according to
            the sampling procedure (i.e., the sampling weights). The array is of shape
            ``(n_coalitions,)``.
        - ``_sampled_coalitions_per_size``: An array with the number of sampled coalitions per size
            (including the empty and full set). The array is of shape ``(n_players + 1,)``.
        - ``_is_coalition_size_sampled``: An array that contains True, if the coalition size was
            sampled and False (computed exactly) otherwise. The array is of shape
            ``(n_players + 1,)``.
        - ``sampled_coalitions_dict``:`` A dictionary containing all sampled coalitions mapping to
            their number of occurrences. The dictionary is of type ``dict[tuple[int, ...], int]``.
    '''
    def __init__(
        self,
        n_players: int,
        sampling_weights: np.ndarray,
        *,
        pairing_trick: bool = True,
        random_state: int | None = None,
        sample_with_replacement: bool = False,
    ) -> None:
        self._n_players = n_players

        if len(sampling_weights) == n_players + 1:
            sampling_weights = sampling_weights[1:-1]
            print('Warning: sampling_weights should be of length n_players-1, ignoring first and last entries.')
        elif len(sampling_weights) == n_players:
            sampling_weights = sampling_weights[1:]
            print('Warning: sampling_weights should be of length n_players-1, ignoring first entry.')
        elif len(sampling_weights) != n_players - 1:
            raise ValueError(f"sampling_weights should be of length n_players-1, but got length {len(sampling_weights)}.")

        self._distribution = sampling_weights / np.min(sampling_weights)
        # Insert 0 for empty coalition size and full coalition size
        self._distribution = np.concatenate(([0.0], self._distribution, [0.0]))

        # Ensure smallest weight is 1
        self._pairing_trick = pairing_trick
        self._rng = np.random.default_rng(seed=random_state)

        self._sampled = False
        self._sample_with_replacement = sample_with_replacement

    def _sampling_probs(self, sizes: np.ndarray) -> np.ndarray:
        '''
        Compute sampling probabilities for given coalition sizes using the constant computed in get_sampling_probs.
        Args:
            sizes (np.ndarray): Array of coalition sizes.
        Returns:
            np.ndarray: Sampling probabilities for the given coalition sizes.
        '''
        return np.minimum(
            self._constant * self._distribution[sizes] / binom(self._n_players, sizes), 1
        )

    def _get_sampling_probs(self, budget: int):
        '''
        Compute sampling probabilities without iteration by inverting the
        piecewise-linear function:
            E(c) = sum_k min(c * weights[k], comb_counts[k])
        where comb_counts[k] = C(n_players, k) and weights[k] = distribution[k].
        For any budget in [0, 2**n_players], this solves for a scale c such that
        E(c) ~= budget (up to floating-point error) and returns sampling_probs(sizes).
        Args:
            budget (int): Total number of coalitions to sample (excluding empty and full coalitions)
        Returns:
            None: Sets self._constant and allows sampling_probs(sizes) to be called.
        (Function written by ChatGPT)
        '''
        n = self._n_players
        sizes = np.arange(1, n)

        # Per-size caps = number of coalitions of that size
        comb_counts = binom(n, sizes).astype(float)          # C(n, k)
        # Per-size weights from the distribution (>= 1 by construction)
        weights = self._distribution[sizes].astype(float)

        # Target expected total, clipped to feasible range [0, 2^n]
        target_total = float(np.clip(budget, 0, np.sum(comb_counts)))
        if target_total == 0.0:
            self._constant = 0.0
            return self._sampling_probs(sizes)

        # Breakpoints where a term saturates: c >= comb_counts[k] / weights[k]
        saturation_thresholds = comb_counts / weights
        order = np.argsort(saturation_thresholds)
        comb_counts_sorted = comb_counts[order]
        weights_sorted = weights[order]
        thresholds_sorted = saturation_thresholds[order]

        # For the segment before saturating index k:
        #   E(c) = sum_{j<k} comb_counts_sorted[j] + c * sum_{j>=k} weights_sorted[j]
        saturated_prefix = np.concatenate(([0.0], np.cumsum(comb_counts_sorted[:-1])))
        weights_prefix = np.concatenate(([0.0], np.cumsum(weights_sorted[:-1])))
        remaining_weight = np.sum(weights_sorted) - weights_prefix

        # Expected total at each breakpoint (just as k would start saturating)
        expected_at_threshold = saturated_prefix + thresholds_sorted * remaining_weight

        # Find the first segment where target_total fits
        segment_idx = np.searchsorted(expected_at_threshold, target_total, side="left")

        if segment_idx >= len(thresholds_sorted):
            # Past all segments: all terms saturate
            scale = float(thresholds_sorted[-1])
        else:
            denom = remaining_weight[segment_idx]
            # If denom == 0, slope is zero (nothing left to grow) -> stick to the threshold
            scale = thresholds_sorted[segment_idx] if denom == 0 else \
                    min((target_total - saturated_prefix[segment_idx]) / denom,
                        thresholds_sorted[segment_idx])

        self._constant = float(scale)

    def _add_one_sample(self, indices: Sequence[int]):
        '''
        Add one sampled coalition to storage.
        Args:
            indices (Sequence[int]): Indices of players in the coalition.
        Returns:
            None: Sample is stored in self._sampled_coalitions_matrix and self._sampledsampled_coalitions_dict
        '''
        self._sampled_coalitions_matrix[self._coalition_idx, indices] = 1
        self._sampledsampled_coalitions_dict[tuple(sorted(indices))] = 1
        self._coalition_idx += 1 

    def sample(self, budget: int):
        '''
        Sample coalitions without replacement according to sampling weights per coalition size.
        Args:
            budget (int): Total number of coalitions to sample (including empty and full coalitions
        Returns:
            None: Samples are stored in self._sampled_coalitions_matrix and self._sampledsampled_coalitions_dict
        '''
        # Budget is an EVEN number between 2 and 2^n
        assert budget >= 2, "Budget must be at least 2"
        budget = min(budget, 2**self._n_players)
        budget += budget % 2

        # Get sampling probabilities
        self._get_sampling_probs(budget-2) # minus 2 for empty and full coalitions
        sizes = np.arange(1, self._n_players)
        samples_per_size = self._symmetric_round_even(
            self._sampling_probs(sizes) * binom(self._n_players, sizes)
        )
        sampling_probs = samples_per_size / binom(self._n_players, sizes)

        # Initialize storage
        self._sampled_coalitions_matrix = np.zeros((budget, self._n_players), dtype=bool)
        self._coalition_idx = 0
        self._sampledsampled_coalitions_dict = {}

        # Sample empty and full coalitions
        self._add_one_sample([])
        self._add_one_sample(list(range(self._n_players)))

        for idx, size in enumerate(sizes):
            if idx >= self._n_players//2 and self._pairing_trick:
                break  # Stop early because of pairing
            if self._pairing_trick and size == self._n_players // 2 and self._n_players % 2 == 0:
                combo_gen = self._combination_generator(
                    self._n_players - 1, size - 1, samples_per_size[idx] // 2
                )
                for indices in combo_gen:
                    self._add_one_sample(list(indices) + [self._n_players - 1])
                    self._add_one_sample(list(set(range(self._n_players-1)) - set(indices)))
            else:
                combo_gen = self._combination_generator(
                    self._n_players, size, samples_per_size[idx]
                )
                for indices in combo_gen:
                    self._add_one_sample(list(indices))
                    if self._pairing_trick:
                        self._add_one_sample(
                            list(set(range(self._n_players)) - set(indices))
                        )

        coalition_sizes = np.sum(self._sampled_coalitions_matrix, axis=1)
        # Assign 1 to sizes of 0 and n
        self._sampled_coalitions_probability = np.ones(self._sampled_coalitions_matrix.shape[0])
        filter_idx = (coalition_sizes > 0) & (coalition_sizes < self._n_players)
        self._sampled_coalitions_probability[filter_idx] = sampling_probs[coalition_sizes[filter_idx]-1]
        self._sampling_adjustment_weights = np.ones(self._sampled_coalitions_matrix.shape[0])
        self._sampling_adjustment_weights[filter_idx] = 1 / sampling_probs[coalition_sizes[filter_idx]-1]

        # Legacy attributes
        self._sampled = True
        self._sampled_coalitions_counter = np.ones(self._sampled_coalitions_matrix.shape[0], dtype=int)
        self._coalition_size_probability = np.minimum(self._sampling_probs(coalition_sizes) * binom(self._n_players, coalition_sizes), 1)

        # Sort out number of coalitions per size
        self._sampled_coalitions_per_size = np.zeros(self._n_players + 1, dtype=int)
        for size in coalition_sizes:
            self._sampled_coalitions_per_size[size] += 1
        self._is_coalition_size_sampled = coalition_sizes > 0
    
    def _symmetric_round_even(self, x: np.ndarray) -> np.ndarray:
        '''
        Given a vector x, returns a vector of integers whose sum is the closest even integer to sum(x),
        and which is symmetric (i.e., the i-th and (n-i)-th entries are the same).
        Args:
            x (np.ndarray): Input vector of floats.
        Returns:
            np.ndarray: Output vector of integers with even sum and symmetry.
        (Function written by ChatGPT)
        '''
        x = np.asarray(x, float); n = x.size
        tgt = int(np.round(x.sum()/2)*2)           # nearest even ≤ sum
        out = np.floor(x).astype(int)
        rem = tgt - out.sum()
        frac = x - np.floor(x)

        pairs = [(i, n-1-i, frac[i]+frac[n-1-i]) for i in range(n//2)]
        pairs.sort(key=lambda t: t[2], reverse=True)
        for i, j, _ in pairs:
            if rem < 2: break
            out[i] += 1; out[j] += 1; rem -= 2
        if n % 2 == 1 and rem == 1:                # give lone +1 to the center
            out[n//2] += 1; rem -= 1
        return out

    def _index_th_combination(self, pool: Sequence[TypeVar("T")], size: int, index: int) -> Tuple[TypeVar("T"), ...]:
        """
        Sample the index-th combination of a given size from the pool in linear time in size of the pool.
        Args:
            pool (Sequence[T]): The pool of elements to choose from.
            size (int): The size of the combination to choose.
            index (int): The index of the combination to return (0-based).
        Returns:
            Tuple[T, ...]: The index-th combination as a tuple.
        (Function written by ChatGPT)
        """
        n = len(pool)
        k = size

        if not (0 <= k <= n):
            raise ValueError("size must be between 0 and len(pool)")
        total = math.comb(n, k)
        if not (0 <= index < total):
            raise IndexError(f"index must be in [0, {total-1}] for C({n},{k})")

        combo = []
        for i in range(n):
            if k == 0:
                break

            # If we must take all remaining items
            if n - i == k:
                combo.extend(pool[i:i+k])
                k = 0
                break

            # Combinations that start by taking pool[i]
            c = math.comb(n - i - 1, k - 1)

            if index < c:
                combo.append(pool[i])
                k -= 1
            else:
                index -= c

        return tuple(combo)

    def _combination_generator(self, n: int, s: int, num_samples: int) -> Sequence[Tuple[int, ...]]:
        '''
        Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
        1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with index_th_combination.
        2. If the number of combinations is large (converting to an int DOES cause an overflow error) OR self._sample_with_replacement is True, randomly sample num_samples combinations directly with replacement.
        Args:
            gen: numpy random generator
            n (int): Size of the pool to sample from.
            s (int): Size of each combination.
            num_samples (int): Number of combinations to sample.
        Yields:
            Tuple[int, ...]: A combination of s elements from the pool of size n.
        '''
        num_combos = math.comb(n, s)
        try:
            assert not self._sample_with_replacement
            indices = self._rng.choice(num_combos, num_samples, replace=False)
            for i in indices:
                yield self._index_th_combination(range(n), s, i)
        except (OverflowError, AssertionError):
            for _ in range(num_samples):
                yield self._rng.choice(n, s, replace=False)