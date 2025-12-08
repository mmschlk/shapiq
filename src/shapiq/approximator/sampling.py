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
        self.n = n_players

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
            self.scale * self.distribution[sizes] / binom(self.n, sizes), 1
        ) 
    
    def get_scale_for_sampling(self, budget: int):
        """
        Compute a scale c such that
            E(c) = sum_k min(c * distribution[k], C(n_players, k)) ~= budget,
        excluding empty and full coalitions.
        Sets self.scale.
        (Function written by ChatGPT)
        """
        n = self.n
        sizes = np.arange(1, n)

        # Number of coalitions per size
        comb_counts = binom(n, sizes).astype(float)           # C(n, k)
        # Per-size weights (must be non-negative)
        weights = self.distribution[sizes].astype(float)

        # Sanity: no negative weights
        if np.any(weights < 0):
            raise ValueError("distribution contains negative entries; scale solving assumes weights >= 0.")

        # Max feasible expected total (#non-empty, non-full subsets)
        max_total = float(np.sum(comb_counts))

        # Clip budget to feasible range
        target_total = float(np.clip(budget, 0, max_total))

        if target_total <= 0.0:
            self.scale = 0.0
            return self.get_sampling_probs(sizes)

        # Helper: E(c)
        def expected_total(c: float) -> float:
            # min(c * w_k, comb_k) summed over k
            return np.minimum(c * weights, comb_counts).sum()

        # --- Find an upper bound where E(c_hi) >= target_total ---
        total_weight = float(weights.sum())

        # If all weights are zero, nothing can grow; scale doesn't matter.
        if total_weight <= 0.0:
            self.scale = 0.0
            return self.get_sampling_probs(sizes)

        # A reasonable first guess if nothing saturates:
        # E(c) ~= c * sum(weights) => c ~= budget / sum(weights)
        c_hi = target_total / total_weight

        # Make sure c_hi is not absurdly tiny
        if c_hi <= 0.0:
            c_hi = 1.0

        # Grow c_hi until E(c_hi) >= target_total (or we hit a safety cap)
        if expected_total(c_hi) < target_total:
            while expected_total(c_hi) < target_total and c_hi < 1e12:
                c_hi *= 2.0

        c_lo = 0.0

        # --- Binary search for c ---
        for _ in range(60):  # ~ 2^-60 relative error; plenty for double precision
            c_mid = 0.5 * (c_lo + c_hi)
            if expected_total(c_mid) < target_total:
                c_lo = c_mid
            else:
                c_hi = c_mid

        scale = c_hi
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
        rem = int(tgt) - int(out.sum())
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
        budget = min(budget, 2**self.n)
        budget += budget % 2

        # Get sampling probabilities
        self.get_scale_for_sampling(budget-2) # Exclude empty and full coalitions from budget
        sizes = np.arange(1, self.n)
        self.samples_per_size = self.symmetric_round_even(
            self.get_sampling_probs(sizes) * binom(self.n, sizes)
        )

        # Initialize storage
        self.coalitions_matrix = np.zeros((budget, self.n), dtype=bool)
        self._coalition_idx = 0
        self.sampled_coalitions_dict = {}

        # Sample empty and full coalitions
        self.add_one_sample([])
        self.add_one_sample(list(range(self.n)))

        for idx, size in enumerate(sizes):
            if idx >= self.n//2 and self.pairing_trick:
                break  # Stop early because of pairing
            if self.pairing_trick and size == self.n // 2 and self.n % 2 == 0:
                combo_gen = self.combination_generator(
                    self.n - 1, size - 1, self.samples_per_size[idx] // 2
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices) + [self.n - 1])
                    self.add_one_sample(list(set(range(self.n-1)) - set(indices)))
            else:
                combo_gen = self.combination_generator(
                    self.n, size, self.samples_per_size[idx]
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices))
                    if self.pairing_trick:
                        self.add_one_sample(
                            list(set(range(self.n)) - set(indices))
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
        coalitions_count = np.zeros(self.n + 1, dtype=int)
        for size in self.coalitions_size:
            coalitions_count[size] += 1
        return coalitions_count

    @property
    def is_coalition_size_sampled(self) -> np.ndarray:
        """
        Returns:
            The Boolean array whether the coalition size was sampled ``(n_players + 1,)``
        """
        is_size_sampled = np.zeros(self.n + 1, dtype=bool)
        is_size_sampled[0] = is_size_sampled[self.n] = True
        is_size_sampled[1:-1] = (self.samples_per_size != binom(self.n, np.arange(1, self.n)))
        return is_size_sampled
    
    @property
    def is_coalition_sampled(self) -> np.ndarray:
        """
        Returns:
            A dictionary indicating whether each coalition was sampled ``(n_coalitions,)``
        """
        return self.is_coalition_size_sampled[self.coalitions_size]

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
    def coalitions_in_size_probability(self) -> np.ndarray:
        """
        Returns:
            The probability a coalition is sampled conditioned on its size ``(n_coalitions,)``
        """
        prob_coalition_per_size = 1 / binom(self.n, np.arange(0, self.n+1))
        return prob_coalition_per_size[self.coalitions_size]
    
    @property
    def coalitions_size_probability(self) -> np.ndarray:
        """
        Returns:
            The probability a coalition size is sampled ``(n_coalitions,)``
        """
        return self.coalitions_probability / self.coalitions_in_size_probability

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
                return int(np.where(self.coalitions_size == self.n)[0][0])
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
    