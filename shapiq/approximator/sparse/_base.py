from collections.abc import Callable
from .._base import Approximator
from ...interaction_values import InteractionValues

import numpy as np

class Sparse(Approximator):

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        top_order: bool,
        sampling_weights: np.ndarray[float] | None = None, # TODO JK Probably useless for us
        random_state: int | None = None,
        transform_type: str = "fourier", # TODO JK: New parameter, to implement fourier or mobius transform
    ) -> None:
        #TODO Implement
        pass


    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        """Approximates the interaction values using a sparse transform approach.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.

        Returns:
            The approximated Shapley interaction values.
        """
        #TODO Implement
        pass




