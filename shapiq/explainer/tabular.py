"""Tabular Explainer class for the shapiq package."""

import warnings
from typing import Optional, Union

import numpy as np

from shapiq.approximator import (
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    RegressionFSII,
    UnbiasedKernelSHAP,
)
from shapiq.approximator._base import Approximator
from shapiq.explainer._base import Explainer
from shapiq.games.imputer import ConditionalImputer, MarginalImputer
from shapiq.interaction_values import InteractionValues

APPROXIMATOR_CONFIGURATIONS = {
    "regression": {
        "SII": InconsistentKernelSHAPIQ,
        "FSII": RegressionFSII,
        "k-SII": InconsistentKernelSHAPIQ,
        "SV": KernelSHAP,
    },
    "permutation": {
        "SII": PermutationSamplingSII,
        "STII": PermutationSamplingSTII,
        "k-SII": PermutationSamplingSII,
        "SV": PermutationSamplingSV,
    },
    "montecarlo": {
        "SII": SHAPIQ,
        "STII": SHAPIQ,
        "FSII": SHAPIQ,
        "k-SII": SHAPIQ,
        "SV": UnbiasedKernelSHAP,
    },
}

AVAILABLE_INDICES = {"SII", "k-SII", "STII", "FSII", "SV"}


class TabularExplainer(Explainer):
    """The tabular explainer as the main interface for the shapiq package.

    The ``TabularExplainer`` class is the main interface for the ``shapiq`` package. It can be used
    to explain the predictions of a model by estimating the Shapley interaction values.

    Args:
        model: The model to be explained as a callable function expecting data points as input and
            returning 1-dimensional predictions.
        data: A background dataset to be used for imputation.
        imputer: Either an object of class Imputer or a string from ``["marginal", "conditional"]``.
            Defaults to ``"marginal"``, which innitializes the default MarginalImputer.
        approximator: An approximator object to use for the explainer. Defaults to ``"auto"``, which will
            automatically choose the approximator based on the number of features and the number of
            samples in the background data.
        index: Type of Shapley interaction index to use. Must be one of ``"SII"`` (Shapley Interaction Index),
            ``"k-SII"`` (k-Shapley Interaction Index), ``"STII"`` (Shapley-Taylor Interaction Index),
            ``"FSII"`` (Faithful Shapley Interaction Index), or ``"SV"`` (Shapley Value) for ``max_order=1``.
            Defaults to ``"k-SII"``.
        max_order: The maximum interaction order to be computed. Defaults to ``2``.
        random_state: The random state to initialize Imputer and Approximator with. Defaults to ``None``.
        **kwargs: Additional keyword-only arguments passed to the imputer.

    Attributes:
        index: Type of Shapley interaction index to use.
        data: A background data to use for the explainer.
        baseline_value: A baseline value of the explainer.
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        imputer="marginal",
        approximator: Union[str, Approximator] = "auto",
        index: str = "k-SII",
        max_order: int = 2,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        if index not in AVAILABLE_INDICES:
            raise ValueError(f"Invalid index `{index}`. " f"Valid indices are {AVAILABLE_INDICES}.")

        super().__init__(model, data)

        self._random_state = random_state
        if imputer == "marginal":
            self._imputer = MarginalImputer(
                self.predict, self.data, random_state=random_state, **kwargs
            )
        elif imputer == "conditional":
            self._imputer = ConditionalImputer(
                self.predict, self.data, random_state=random_state, **kwargs
            )
        elif isinstance(imputer, MarginalImputer) or isinstance(imputer, ConditionalImputer):
            self._imputer = imputer
        else:
            raise ValueError(
                f"Invalid imputer {imputer}. "
                f'Must be one of ["marginal", "conditional"], or a valid Imputer object.'
            )
        self._n_features: int = self.data.shape[1]

        self.index = index
        self._max_order: int = max_order
        self._approximator = self._init_approximator(approximator, self.index, self._max_order)

    def explain(
        self, x: np.ndarray, budget: Optional[int] = None, random_state: Optional[int] = None
    ) -> InteractionValues:
        """Explains the model's predictions.

        Args:
            x: The data point to explain as a 2-dimensional array with shape
                (1, n_features).
            budget: The budget to use for the approximation. Defaults to `None`, which will
                set the budget to 2**n_features based on the number of features.
            random_state: The random state to re-initialize Imputer and Approximator with. Defaults to ``None``.
        """
        if budget is None:
            budget = 2**self._n_features
            if budget > 2048:
                warnings.warn(
                    f"Using the budget of 2**n_features={budget}, which might take long\
                              to compute. Set the `budget` parameter to suppress this warning."
                )
        if random_state is not None:
            self._imputer._rng = np.random.default_rng(random_state)
            self._approximator._rng = np.random.default_rng(random_state)
            self._approximator._sampler._rng = np.random.default_rng(random_state)

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
            if max_order == 1:
                if index != "SV":
                    warnings.warn(
                        "`max_order=1` but `index != 'SV'`, setting `index = 'SV'`. Using the KernelSHAP approximator."
                    )
                    self.index = "SV"
                return KernelSHAP(
                    n=self._n_features,
                    random_state=self._random_state,
                )
            elif index == "SV":
                if max_order != 1:
                    warnings.warn(
                        "`index='SV'` but `max_order != 1`, setting `max_order = 1`. Using the KernelSHAP approximator."
                    )
                    self._max_order = 1
                return KernelSHAP(
                    n=self._n_features,
                    random_state=self._random_state,
                )
            elif index == "FSII":
                return RegressionFSII(
                    n=self._n_features,
                    max_order=max_order,
                    random_state=self._random_state,
                )
            elif index == "SII" or index == "k-SII":
                return KernelSHAPIQ(
                    n=self._n_features,
                    max_order=max_order,
                    random_state=self._random_state,
                    index=index,
                )
            else:
                return SVARMIQ(
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
