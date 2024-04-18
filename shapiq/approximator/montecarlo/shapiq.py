"""SHAP-IQ approximation."""

from typing import Optional

from ._base import MonteCarlo


class SHAPIQ(MonteCarlo):
    """SHAP-IQ approximator uses standard form of Shapley interactions.
    Algorithm described in https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html.
    This is the default method from MonteCarlo approximator with no stratification.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index
        random_state: The random state of the estimator. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, min_order
            is equal to 1.
        iteration_cost: The cost of a single iteration of the regression SII.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str = "k-SII",
        top_order: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n,
            max_order,
            index=index,
            top_order=top_order,
            stratify_coalition_size=False,
            stratify_intersection_size=False,
            random_state=random_state,
        )
