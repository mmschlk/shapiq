"""This module contains stochastic sampling procedures for coalitions of players."""
import copy

import numpy as np
from typing import Optional
from scipy.special import binom
from utils import powerset
import warnings
from collections import Counter

class CoalitionSampler:
    """The coalition sampler to generate a collection of subsets as a basis for approximation methods.

    Args:
        n: Number of elements to sample from
        sampling_weights: Sampling for weights for coalition sizes, must be non-negative and at least one >0.
        random_state: The random state to use for the sampling process. Defaults to `None`.

    Attributes:
        budget: Sampling budget, i.e. number of distinct subset that will be sampled
        sampling_weights: Sampling for weights for coalition sizes.
        random_state: The random state to use for the sampling process. Defaults to `None`.

    Example:
    """

    def __init__(
        self,
         n: np.ndarray,
         sampling_weights: np.ndarray,
         pairing_trick: bool = False,
         random_state: Optional[int] = None
    ) -> None:
        self.n = n
        self.N = np.array(range(n))
        self.sampling_weights = sampling_weights/np.sum(sampling_weights)
        self.pairing_trick = pairing_trick
        self._random_state = random_state
        self._rng: Optional[np.random.Generator] = np.random.default_rng(seed=self._random_state)
        self.coalitions_to_exclude = []
        self.coalitions_to_compute = []
        self.coalitions_to_sample = list(range(self.n+1))
        #Set maximum of possible coalitions by excluding zero-weighted coalition sizes
        # Coalition sizes with zero weights, should be removed and excluded from total number of coalitions
        self.n_max_coalitions = 2 ** self.n
        #Handle zero-weighted coalition sizes
        for size, weight in enumerate(self.sampling_weights):
            if weight == 0:
                self.n_max_coalitions -= int(binom(self.n, size))
                self.coalitions_to_exclude.extend([self.coalitions_to_sample.pop(self.coalitions_to_sample.index(size))])
        self.adjusted_sampling_weights = self.sampling_weights[self.coalitions_to_sample]/np.sum(self.sampling_weights[self.coalitions_to_sample])
        #Check shape of sampling weights for coalition sizes 0,...,n
        assert n+1==np.size(sampling_weights), "n elements must correspond to n+1 coalition sizes (including empty subsets)"
        #Check non-negativity of sampling weights
        assert (sampling_weights>=0).all(),"All sampling weights must be non-negative"

    def get_coalitions_matrix(self):
        return copy.deepcopy(self.sampled_coalitions_matrix)

    def get_coalitions_counter(self):
        return copy.deepcopy(self.sampled_coself.coalitions_to_samplealitions_counter)

    def get_coalitions_probabilities(self):
        return copy.deepcopy(self.sampled_coalitions_probs)

    def _execute_border_trick(self,sampling_budget):
        coalitions_per_size = np.array([binom(self.n, k) for k in range(self.n + 1)])
        expected_number_of_coalitions = sampling_budget * self.adjusted_sampling_weights
        sampling_exceeds_expectation = expected_number_of_coalitions >= coalitions_per_size[self.coalitions_to_sample]
        while sampling_exceeds_expectation.any():
            coalitions_to_move = [self.coalitions_to_sample[index] for index, include in enumerate(sampling_exceeds_expectation) if include]
            self.coalitions_to_compute.extend(
                [self.coalitions_to_sample.pop(self.coalitions_to_sample.index(move_this)) for move_this in coalitions_to_move])
            sampling_budget -= int(np.sum(coalitions_per_size[coalitions_to_move]))
            self.adjusted_sampling_weights = self.adjusted_sampling_weights[~sampling_exceeds_expectation] / np.sum(
                self.adjusted_sampling_weights[~sampling_exceeds_expectation])
            expected_number_of_coalitions = sampling_budget * self.adjusted_sampling_weights
            sampling_exceeds_expectation = expected_number_of_coalitions >= coalitions_per_size[self.coalitions_to_sample]
        return sampling_budget

    def sample(
    self,
    sampling_budget:int
    ) -> np.ndarray:
        """Samples distinct coalitions according to the specified budget.

        Args:
        budget: The budget for the approximation (i.e., the number of distinct coalitions).

        Returns:
        A binary matrix of coalitions normalized with probabilities and number of Monte Carlo samples.
        A dictionary of parameters of the sampling procedures, including:
        - the number of occurrences of each coalition
        - the probability of a coalition to be included in the collection
        - a list indicating if the coalitions were sampled or computed explicitly (the latter is only applicable, if border_trick = True)
        - the number of coalitions for each coalition size
        """

        self.sampled_coalitions_dict = {}


        if sampling_budget >= self.n_max_coalitions:
            warnings.warn("Not all budget is required due to the border-trick.", UserWarning)

        #Adjust sampling budget to max coalitions
        sampling_budget = min(sampling_budget,self.n_max_coalitions)

        #Variables to be computed
        self.coalitions_per_size = np.zeros(self.n+1,dtype=int)
        self.sampled_coalitions_counter = np.zeros((sampling_budget))
        self.sampled_coalitions_matrix = np.zeros((sampling_budget, self.n), dtype=int)
        self.sampled_coalitions_probs = np.zeros((sampling_budget), dtype=float)
        self.is_coalition_size_sampled = np.zeros(self.n+1,dtype=bool)
        permutation = np.arange(self.n)

        #Border-Trick: Enumerate all coalitions, where the expected number of coalitions exceeds the total number
        sampling_budget = self._execute_border_trick(sampling_budget)

        #Sort by size for esthetics
        self.coalitions_to_sample.sort()
        self.coalitions_to_compute.sort()
        if len(self.coalitions_to_sample) > 0: #Sample, if there are coalition sizes to sample from
            #Draw random coalition sizes based on adjusted_sampling_weights
            coalition_sizes = self._rng.choice(self.coalitions_to_sample, p=self.adjusted_sampling_weights,size=sampling_budget)
            initial_budget = len(coalition_sizes)
            counter = 0
            #Start sampling procedure
            while sampling_budget > 0:
                #generate random permutation
                self._rng.shuffle(permutation)
                coalition_size = coalition_sizes[counter]
                counter += 1
                counter %= initial_budget
                coalition_indices = permutation[:coalition_size]
                coalition_indices.sort()  # Sorting for consistent representation
                coalition_tuple = tuple(coalition_indices)
                self.coalitions_per_size[coalition_size] += 1
                if self.sampled_coalitions_dict.get(coalition_tuple):
                    #if coalition is not new
                    self.sampled_coalitions_dict[coalition_tuple] += 1
                else:
                    self.sampled_coalitions_dict[coalition_tuple] = 1
                    sampling_budget -= 1
                if self.pairing_trick and sampling_budget > 0:
                    paired_coalition_size = self.n-coalition_size
                    if paired_coalition_size not in self.coalitions_to_sample:
                        warnings.warn("Pairing is affected as weights are not symmetric and border-trick is enabled.",UserWarning)
                    else:
                        paired_coalition_indices = permutation[coalition_size:]
                        paired_coalition_indices.sort()
                        paired_coalition_tuple = tuple(paired_coalition_indices)
                        self.coalitions_per_size[paired_coalition_size] += 1
                        if self.sampled_coalitions_dict.get(paired_coalition_tuple):
                            #if coalition is not new
                            self.sampled_coalitions_dict[paired_coalition_tuple] += 1
                        else:
                            self.sampled_coalitions_dict[paired_coalition_tuple] = 1
                            sampling_budget -= 1


        # Convert coalition counts to a binary matrix
        coalition_index = 0
        for coalition_size in self.coalitions_to_compute:
            self.coalitions_per_size[coalition_size] = int(binom(self.n,coalition_size))
            for T in powerset(range(n),min_size=coalition_size,max_size=coalition_size):
                self.sampled_coalitions_matrix[coalition_index, list(T)] = 1
                self.sampled_coalitions_counter[coalition_index] = 1
                self.sampled_coalitions_probs[coalition_index] = 1
                coalition_index += 1
        for coalition_tuple, count in self.sampled_coalitions_dict.items():
            self.sampled_coalitions_matrix[coalition_index, list(coalition_tuple)] = 1
            self.sampled_coalitions_counter[coalition_index] = count
            self.sampled_coalitions_probs[coalition_index] = self.adjusted_sampling_weights[self.coalitions_to_sample.index(len(coalition_tuple))]/self.coalitions_per_size[len(coalition_tuple)]
            coalition_index += 1

        for coalition_size in self.coalitions_to_sample:
            self.is_coalition_size_sampled[coalition_size] = True


if __name__ == "__main__":
    n = 10
    sampling_weights = np.zeros(n+1)
    sampling_weights[4:6] = 1
    #sampling_weights[1] = -1
    #sampling_weights[4:6] *= 1000000
    sampler = CoalitionSampler(n,sampling_weights,pairing_trick=False,random_state=43)
    sampler.sample(5000)
    coalitions_matrix = sampler.get_coalitions_matrix()
    print("ok")