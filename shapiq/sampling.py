"""This module contains stochastic sampling procedures for coalitions of players."""
import numpy as np
from typing import Optional
from scipy.special import binom
from utils import powerset

class CoalitionSampler():
    """The coalition sampler to generate a collection of subsets as a basis for approximation methods.

    Args:
        n: Number of elements to sample from
        sampling_weights: Sampling for weights for coalition sizes.
        random_state: The random state to use for the sampling process. Defaults to `None`.

    Attributes:
        budget: Sampling budget, i.e. number of distinct subset that will be sampled
        sampling_weights: Sampling for weights for coalition sizes.
        random_state: The random state to use for the sampling process. Defaults to `None`.

    Example:
        >>> from games import DummyGame
        >>> from approximator import ShapIQ
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = ShapIQ(n=5, max_order=2, index="SII")
        >>> approximator.approximate(budget=50, game=game)
        InteractionValues(
            index=SII, order=2, estimated=False, estimation_budget=32,
            values={
                (0,): 0.2,
                (1,): 0.7,
                (2,): 0.7,
                (3,): 0.2,
                (4,): 0.2,
                (0, 1): 0,
                (0, 2): 0,
                (0, 3): 0,
                (0, 4): 0,
                (1, 2): 1.0,
                (1, 3): 0,
                (1, 4): 0,
                (2, 3): 0,
                (2, 4): 0,
                (3, 4): 0
            }
        )
    """

    def __init__(
        self,
         n: np.ndarray,
         sampling_weights: np.ndarray,
         border_trick: bool = False,
         pairing_trick: bool = False,
         random_state: Optional[int] = None
    ) -> None:
        self.n = n
        self.sampling_weights = sampling_weights/np.sum(sampling_weights)
        self.border_trick = border_trick
        self.pairing_trick = pairing_trick
        self.random_state = random_state
        assert n+1==np.size(sampling_weights), "n elements corresponds to n+1 subset sizes (including empty subsets)"

    def sample_coalitions_from_budget(
    self,
    budget:int
    ) -> np.ndarray:
        """Samples distinct coalitions according to the specified budget.

        Args:
        budget: The budget for the approximation (i.e., the number of distinct coalitions).

        Returns:
        A binary matrix of coalitions normalized with probabilities and number of Monte Carlo samples.
        """

        sampling_budget = budget
        adjusted_sampling_weights = self.sampling_weights/np.sum(self.sampling_weights)
        sampled_coalitions_counter_dict = {}
        sampled_coalitions_counter = np.zeros((sampling_budget))
        sampled_coalitions_matrix = np.zeros((sampling_budget, self.n), dtype=int)
        sampled_coalitions_weights = np.zeros((sampling_budget),dtype=float)
        coalitions_to_compute = []
        coalitions_to_sample = list(range(self.n+1))

        if self.border_trick:
            coalitions_per_size = np.array([binom(self.n,k) for k in range(n+1)])
            expected_number_of_coalitions = budget * self.sampling_weights
            sampling_exceeds_expectation = expected_number_of_coalitions>=coalitions_per_size
            while sampling_exceeds_expectation.any():
                coalitions_to_move = [index for index, include in enumerate(sampling_exceeds_expectation) if include]
                coalitions_to_compute.extend([coalitions_to_sample.pop(index) for index in sorted(coalitions_to_move,reverse=True)])
                sampling_budget -= np.sum(coalitions_per_size[coalitions_to_move])
                adjusted_sampling_weights = adjusted_sampling_weights[~sampling_exceeds_expectation]/np.sum(adjusted_sampling_weights[~sampling_exceeds_expectation])
                expected_number_of_coalitions = sampling_budget*adjusted_sampling_weights
                sampling_exceeds_expectation = expected_number_of_coalitions >= coalitions_per_size[coalitions_to_sample]
                if len(coalitions_to_sample) == 0 and sampling_budget > 0:
                    sampled_coalitions_matrix = np.zeros((2**self.n, self.n), dtype=int)
                    sampled_coalitions_counter = np.zeros(2**self.n)
                    sampled_coalitions_weights = np.zeros(2**self.n)

                    Warning("Not all budget might be needed due to the border-trick")
                    break

        coalitions_to_sample.sort()
        coalitions_to_compute.sort()
        if len(coalitions_to_sample) > 0:
            #Start sampling procedure
            while sampling_budget > 0:
                coalition_size = np.random.choice(coalitions_to_sample, p=adjusted_sampling_weights)
                coalition_indices = np.random.choice(self.n, size=coalition_size, replace=True)
                coalition_indices.sort()  # Sorting for consistent representation
                coalition_tuple = tuple(coalition_indices)
                if sampled_coalitions_counter_dict.get(coalition_tuple):
                    #if coalition is not new
                    sampled_coalitions_counter_dict[coalition_tuple] += 1
                else:
                    sampled_coalitions_counter_dict[coalition_tuple] = 1
                    sampling_budget -= 1
                if self.pairing_trick and sampling_budget > 0:
                    paired_coalition_size = self.n-coalition_size
                    if paired_coalition_size not in coalitions_to_sample:
                        Warning("Pairing is affected as weights are not symmetric and border-trick is enabled.")
                    else:
                        paired_coalition_tuple = tuple(set(range(self.n))-set(coalition_tuple))
                        if sampled_coalitions_counter_dict.get(paired_coalition_tuple):
                            #if coalition is not new
                            sampled_coalitions_counter_dict[paired_coalition_tuple] += 1
                        else:
                            sampled_coalitions_counter_dict[paired_coalition_tuple] = 1
                            sampling_budget -= 1


        # Convert coalition counts to a binary matrix
        coalition_index = 0
        for coalition_size in coalitions_to_compute:
            for T in powerset(range(n),min_size=coalition_size,max_size=coalition_size):
                sampled_coalitions_matrix[coalition_index, list(T)] = 1
                sampled_coalitions_counter[coalition_index] = 1
                sampled_coalitions_weights[coalition_index] = 1
                coalition_index+=1
        for coalition_tuple, count in sampled_coalitions_counter.items():
            sampled_coalitions_matrix[coalition_index, list(coalition_tuple)] = 1
            sampled_coalitions_counter[coalition_index] = count
            sampled_coalitions_weights[coalition_index] = 1
            coalition_index += 1

        return sampled_coalitions_matrix


if __name__ == "__main__":
    n = 10
    sampler = CoalitionSampler(n,np.ones(n+1),border_trick=True,pairing_trick=True)
    result = sampler.sample_coalitions_from_budget(500)