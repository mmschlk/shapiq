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
    
    Uses sampling method from Musco and Witter (2025) "Provably Accurate Shapley Value Estimation via Leverage Score Sampling"
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
        self.n_players = n_players

        if len(sampling_weights) == n_players + 1:
            sampling_weights = sampling_weights[1:-1]
            print('Warning: sampling_weights should be of length n_players-1, ignoring first and last entries.')
        elif len(sampling_weights) == n_players:
            sampling_weights = sampling_weights[1:]
            print('Warning: sampling_weights should be of length n_players-1, ignoring first entry.')
        elif len(sampling_weights) != n_players - 1:
            raise ValueError(f"sampling_weights should be of length n_players-1, but got length {len(sampling_weights)}.")

        self.distribution = sampling_weights / np.min(sampling_weights)
        # Insert 0 for empty coalition size and full coalition size
        self.distribution = np.concatenate(([0.0], self.distribution, [0.0]))

        self.pairing_trick = pairing_trick
        self.sample_with_replacement = sample_with_replacement
        self.set_random_state(random_state)

    def get_sampling_probs(self, sizes: np.ndarray) -> np.ndarray:
        '''
        Compute sampling probabilities for given coalition sizes using the scale computed in get_scale_for_sampling.
        Args:
            sizes (np.ndarray): Array of coalition sizes.
        Returns:
            np.ndarray: Sampling probabilities for the given coalition sizes.
        '''
        return np.minimum(
            self.scale * self.distribution[sizes] / binom(self.n_players, sizes), 1
        )

    def get_scale_for_sampling(self, budget: int):
        '''
        Compute sampling probabilities without iteration by inverting the
        piecewise-linear function:
            E(c) = sum_k min(c * distribution[k], choose(n_players, k))
        For any budget in [0, 2**n_players], this solves for a scale c such that
        E(c) ~= budget (up to floating-point error).
        Args:
            budget (int): Total number of coalitions to sample (excluding empty and full coalitions)
        Returns:
            None: Sets self.scale so that self.get_sampling_probs(sizes) gives correct probabilities.
        (Function written by ChatGPT)
        '''
        n = self.n_players
        sizes = np.arange(1, n)

        # Per-size caps = number of coalitions of that size
        comb_counts = binom(n, sizes).astype(float)          # C(n, k)
        # Per-size weights from the distribution (>= 1 by construction)
        weights = self.distribution[sizes].astype(float)

        # Target expected total, clipped to feasible range [0, 2^n]
        target_total = float(np.clip(budget, 0, np.sum(comb_counts)))
        if target_total == 0.0:
            self.scale = 0.0
            return self.get_sampling_probs(sizes)

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

        self.scale = float(scale)

    def add_one_sample(self, indices: Sequence[int]):
        '''
        Add one sampled coalition to storage.
        Args:
            indices (Sequence[int]): Indices of players in the coalition.
        Returns:
            None: Sample is stored in self.coalitions_matrix and self.sampled_coalitions_dict
        '''
        self.coalitions_matrix[self._coalition_idx, indices] = 1
        if tuple(sorted(indices)) not in self.sampled_coalitions_dict:
            self.sampled_coalitions_dict[tuple(sorted(indices))] = 0
        self.sampled_coalitions_dict[tuple(sorted(indices))] += 1
        self._coalition_idx += 1 

    def symmetric_round_even(self, x: np.ndarray) -> np.ndarray:
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

    def index_th_combination(self, pool: Sequence[TypeVar("T")], size: int, index: int) -> Tuple[TypeVar("T"), ...]:
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

    def combination_generator(self, n: int, s: int, num_samples: int) -> Sequence[Tuple[int, ...]]:
        '''
        Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
        1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with index_th_combination.
        2. If the number of combinations is large (converting to an int DOES cause an overflow error) OR self.sample_with_replacement is True, randomly sample num_samples combinations directly with replacement.
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
            assert not self.sample_with_replacement
            indices = self._rng.choice(num_combos, num_samples, replace=False)
            for i in indices:
                yield self.index_th_combination(range(n), s, i)
        except (OverflowError, AssertionError):
            for _ in range(num_samples):
                yield self._rng.choice(n, s, replace=False)

    def sample(self, budget: int):
        '''
        Sample coalitions according to sampling weights per coalition size.
        Args:
            budget (int): Total number of coalitions to sample (including empty and full coalitions)
        Returns:
            None: Samples are stored in self.coalitions_matrix and self.sampled_coalitions_dict
        '''
        # Budget is an EVEN number between 2 and 2^n
        assert budget >= 2, "Budget must be at least 2"
        budget = min(budget, 2**self.n_players)
        budget += budget % 2

        # Get sampling probabilities
        self.get_scale_for_sampling(budget-2) # Exclude empty and full coalitions from budget
        sizes = np.arange(1, self.n_players)
        samples_per_size = self.symmetric_round_even(
            self.get_sampling_probs(sizes) * binom(self.n_players, sizes)
        )

        # Initialize storage
        self.coalitions_matrix = np.zeros((budget, self.n_players), dtype=bool)
        self._coalition_idx = 0
        self.sampled_coalitions_dict = {}

        # Sample empty and full coalitions
        self.add_one_sample([])
        self.add_one_sample(list(range(self.n_players)))

        for idx, size in enumerate(sizes):
            if idx >= self.n_players//2 and self.pairing_trick:
                break  # Stop early because of pairing
            if self.pairing_trick and size == self.n_players // 2 and self.n_players % 2 == 0:
                combo_gen = self.combination_generator(
                    self.n_players - 1, size - 1, samples_per_size[idx] // 2
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices) + [self.n_players - 1])
                    self.add_one_sample(list(set(range(self.n_players-1)) - set(indices)))
            else:
                combo_gen = self.combination_generator(
                    self.n_players, size, samples_per_size[idx]
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices))
                    if self.pairing_trick:
                        self.add_one_sample(
                            list(set(range(self.n_players)) - set(indices))
                        )

    @property
    def n_coalitions(self) -> int:
        """
        Returns:
            The number of coalitions that have been sampled.
        """
        try:
            return int(self.coalitions_matrix.shape[0])
        except AttributeError:  # if not sampled
            return 0

    @property
    def coalitions_size(self) -> np.ndarray:
        """Returns the coalition sizes of the sampled coalitions.

        Returns:
            The coalition sizes of the sampled coalitions.

        """
        return np.sum(self.coalitions_matrix, axis=1)
    
    @property
    def coalitions_per_size(self) -> np.ndarray:
        """
        Returns:
            An array with the number of coalitions sampled per coalition size ``(n_players + 1,)``
        """
        coalitions_count = np.zeros(self.n_players + 1, dtype=int)
        for size in self.coalitions_size:
            coalitions_count[size] += 1
        return coalitions_count

    @property
    def is_coalition_size_sampled(self) -> np.ndarray:
        """
        Returns:
            The Boolean array whether the coalition size was sampled ``(n_players + 1,)``
        """
        is_size_sampled = np.zeros(self.n_players + 1, dtype=bool)
        is_size_sampled[self.coalitions_size] = True
        return is_size_sampled
    
    @property
    def is_coalition_sampled(self) -> np.ndarray:
        """
        Returns:
            A dictionary indicating whether each coalition was sampled ``(n_coalitions,)``
        """
        return self.is_coalition_sampled[self.coalitions_size]

    @property
    def coalitions_probability(self) -> np.ndarray:
        """
        Returns:
            A copy of the sampled coalitions probabilities of shape ``(n_coalitions,)``
        """
        probs = self.get_sampling_probs(self.coalitions_size)
        # Replace the empty and full coalition probabilities with 1
        probs[self.empty_coalition_index] = 1.0
        probs[self.full_coalition_index] = 1.0
        return probs

    @property
    def sampling_adjustment_weights(self) -> np.ndarray:
        """
        Returns:
            An array with adjusted weight for each coalition ``(n_coalitions,)``
        """
        return 1 / self.coalitions_probability
    
    @property
    def coalitions_counter(self) -> np.ndarray:
        """
        Returns:
            An array with the number of times each coalition was sampled ``(n_coalitions,)``
        """
        # Iterate over each coalition in the coalitions_matrix and get its count from sampled_coalitions_dict
        counts = np.zeros(self.n_coalitions, dtype=int)
        for i in range(self.n_coalitions):
            coalition_tuple = tuple(np.where(self.coalitions_matrix[i])[0])
            counts[i] = self.sampled_coalitions_dict.get(coalition_tuple, 0)
        return counts

    @property
    def empty_coalition_index(self) -> int | None:
        """
        Returns:
            The index of the empty coalition or ``None`` if the empty coalition was not sampled.
        """
        try:
            if self.coalitions_per_size[0] >= 1:
                return int(np.where(self.coalitions_size == 0)[0][0])
        except IndexError:
            pass
        return None

    @property
    def full_coalition_index(self) -> int | None:
        """
        Returns:
            The index of the full coalition or ``None`` if the full coalition was not sampled.
        """
        try:
            if self.coalitions_per_size[-1] >= 1:
                return int(np.where(self.coalitions_size == self.n_players)[0][0])
        except IndexError:
            pass
        return None
    
    def set_random_state(self, random_state: int | None) -> None:
        '''
        Set the random state of the sampler.
        Args:
            random_state (int | None): Random seed for reproducibility
        '''
        self._rng = np.random.default_rng(seed=random_state)
    