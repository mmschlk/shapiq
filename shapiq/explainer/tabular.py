"""This module contains the interaction explainer for the shapiq package. This is the main interface
for users of the shapiq package."""

import warnings
from typing import Optional, Union

import numpy as np

from shapiq.approximator import (
    PermutationSamplingSII,
    PermutationSamplingSTII,
    RegressionFSII,
    RegressionSII,
    ShapIQ,
)
from shapiq.approximator._base import Approximator
from shapiq.explainer._base import Explainer
from shapiq.games.imputer import MarginalImputer
from shapiq.interaction_values import InteractionValues

APPROXIMATOR_CONFIGURATIONS = {
    "Regression": {"SII": RegressionSII, "FSII": RegressionFSII, "k-SII": RegressionSII},
    "Permutation": {
        "SII": PermutationSamplingSII,
        "STII": PermutationSamplingSTII,
        "kSII": PermutationSamplingSII,
    },
    "ShapIQ": {"SII": ShapIQ, "STII": ShapIQ, "FSII": ShapIQ, "k-SII": ShapIQ},
}

AVAILABLE_INDICES = {"SII", "k-SII", "STII", "FSII"}


class TabularExplainer(Explainer):
    """The tabular explainer as the main interface for the shapiq package.

    The `TabularExplainer` class is the main interface for the `shapiq` package. It can be used
    to explain the predictions of a model by estimating the Shapley interaction values.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        data: A background dataset to use for the explainer.
        approximator: An approximator to use for the explainer. Defaults to `"auto"`, which will
            automatically choose the approximator based on the number of features and the number of
            samples in the background data.
        index: Type of Shapley interaction index to use. Must be one of `"SII"` (Shapley Interaction Index),
            `"k-SII"` (k-Shapley Interaction Index), `"STII"` (Shapley-Taylor Interaction Index), or
            `"FSII"` (Faithful Shapley Interaction Index). Defaults to `"k-SII"`.

    Attributes:
        index: Type of Shapley interaction index to use.
        data: A background data to use for the explainer.
        baseline_value: A baseline value of the explainer.
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        approximator: Union[str, Approximator] = "auto",
        index: str = "k-SII",
        max_order: int = 2,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:

        if index not in AVAILABLE_INDICES:
            raise ValueError(f"Invalid index `{index}`. " f"Valid indices are {AVAILABLE_INDICES}.")
        if max_order < 2:
            raise ValueError("The maximum order must be at least 2.")

        super().__init__(model, data)

        self._random_state = random_state
        self._imputer = MarginalImputer(self.predict, self.data)
        self._n_features: int = self.data.shape[1]

        self.index = index
        self._max_order: int = max_order
        self._approximator = self._init_approximator(approximator, self.index, self._max_order)
        self._rng = np.random.default_rng(self._random_state)

    def explain(self, x: np.ndarray, budget: Optional[int] = None) -> InteractionValues:
        """Explains the model's predictions.

        Args:
            x: The data point to explain as a 2-dimensional array with shape
                (1, n_features).
            budget: The budget to use for the approximation. Defaults to `None`, which will
                set the budget to 2**n_features based on the number of features.
        """
        if budget is None:
            budget = 2**self._n_features
            if budget > 2048:
                warnings.warn(
                    f"Using the budget of 2**n_features={budget}, which might take long\
                              to compute. Set the `budget` parameter to suppress this warning."
                )

        # initialize the imputer with the explanation point
        imputer = self._imputer.fit(x)

        # explain
        interaction_values = self._approximator.approximate(budget=budget, game=imputer)
        interaction_values.baseline_value = self.baseline_value

        return interaction_values

    @property
    def baseline_value(self) -> float:
        """Returns the baseline value of the explainer."""
        return self._imputer.empty_prediction

    def _init_approximator(
        self, approximator: Union[Approximator, str], index: str, max_order: int
    ) -> Approximator:
        if isinstance(approximator, Approximator):  # if the approximator is already given
            return approximator
        if approximator == "auto":
            if index == "FSII":
                return RegressionFSII(
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
            approximator = APPROXIMATOR_CONFIGURATIONS[approximator][index]
        except KeyError:
            raise ValueError(
                f"Invalid approximator `{approximator}` or index `{index}`. "
                f"Valid configuration are described in {APPROXIMATOR_CONFIGURATIONS}."
            )
        # initialize the approximator class with params
        init_approximator = approximator(n=self._n_features, max_order=max_order)
        return init_approximator
