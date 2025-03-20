from collections.abc import Callable
from .._base import Approximator
import numpy as np

class Sparse(Approximator):

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        top_order: bool,
        min_order: int = 0,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray[float] | None = None,
        random_state: int | None = None,
    ) -> None:
        