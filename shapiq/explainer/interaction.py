"""This module contains the interaction explainer for the shapiq package. This is the main interface
for users of the shapiq package."""
from typing import Callable, Union, Optional

import numpy as np

from approximator._base import Approximator
from approximator._interaction_values import InteractionValues
from ._base import Explainer
from approximator import (
    RegressionSII,
    RegressionFSI,
    PermutationSamplingSII,
    PermutationSamplingSTI,
    ShapIQ,
)


__all__ = ["InteractionExplainer"]


APPROXIMATOR_CONFIGURATIONS = {
    "Regression": {"SII": RegressionSII, "FSI": RegressionFSI, "k-SII": RegressionSII},
    "Permutation": {
        "SII": PermutationSamplingSII,
        "STI": PermutationSamplingSTI,
        "kSII": PermutationSamplingSII,
    },
    "ShapIQ": {"SII": ShapIQ, "STI": ShapIQ, "FSI": ShapIQ, "k-SII": ShapIQ},
}

AVAILABLE_INDICES = {
    index
    for approximator_dict in APPROXIMATOR_CONFIGURATIONS.values()
    for index in approximator_dict.keys()
}


class InteractionExplainer(Explainer):
    """The interaction explainer as the main interface for the shapiq package.

    The interaction explainer is the main interface for the shapiq package. It can be used to
    explain the predictions of a model by estimating the Shapley interaction values.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        background_data: The background data to use for the explainer.
        approximator: The approximator to use for the explainer. Defaults to `"auto"`, which will
            automatically choose the approximator based on the number of features and the number of
            samples in the background data.
        index: The Shapley interaction index to use. Must be one of `"SII"` (Shapley Interaction Index),
        `"kSII"` (n-Shapley Interaction Index), `"STI"` (Shapley-Taylor Interaction Index), or
        `"FSI"` (Faithful Shapley Interaction Index). Defaults to `"kSII"`.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        approximator: Union[str, Approximator] = "auto",
        index: str = "k-SII",
        max_order: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(model, background_data)
        if index not in AVAILABLE_INDICES:
            raise ValueError(f"Invalid index `{index}`. " f"Valid indices are {AVAILABLE_INDICES}.")
        self.index = index
        self._default_budget: int = 2_000
        if max_order < 2:
            raise ValueError("The maximum order must be at least 2.")
        self._max_order: int = max_order
        self._random_state = random_state
        self._rng = np.random.default_rng(self._random_state)
        self.approximator = self._init_approximator(approximator, index, max_order)

    def explain(self, x_explain: np.ndarray, budget: Optional[int] = None) -> InteractionValues:
        """Explains the model's predictions.

        Args:
            x_explain: The data point to explain as a 2-dimensional array with shape
                (1, n_features).
            budget: The budget to use for the approximation. Defaults to `None`, which will choose
                the budget automatically based on the number of features.
        """
        if budget is None:
            budget = min(2**self._n_features, self._default_budget)

        # initialize the imputer with the explanation point
        imputer = self._imputer.fit(x_explain)

        # explain
        interaction_values = self.approximator.approximate(budget=budget, game=imputer)

        return interaction_values

    def _init_approximator(
        self, approximator: Union[Approximator, str], index: str, max_order: int
    ) -> Approximator:
        if isinstance(approximator, Approximator):  # if the approximator is already given
            return approximator
        if approximator == "auto":
            if index == "FSI":
                return RegressionFSI(
                    n=self._n_features,
                    max_order=max_order,
                    random_state=self._random_state,
                )
            else:  # default to ShapIQ
                return ShapIQ(
                    n=self._n_features,
                    max_order=max_order,
                    top_order=False,
                    random_state=self._random_state,
                    index=index,
                )
        # assume that the approximator is a string
        try:
            approximator_class = APPROXIMATOR_CONFIGURATIONS[approximator][index]
        except KeyError:
            raise ValueError(
                f"Invalid approximator `{approximator}` or index `{index}`. "
                f"Valid configuration are described in {APPROXIMATOR_CONFIGURATIONS}."
            )
        # initialize the approximator class with params
        init_approximator = approximator_class.__init__(n=self._n_features, max_order=max_order)
        return init_approximator
