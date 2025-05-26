"""Tabular Explainer class for the shapiq package."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import numpy as np

from shapiq.approximator import (
    SHAPIQ,
    SPEX,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    RegressionFBII,
    RegressionFSII,
    UnbiasedKernelSHAP,
)
from shapiq.approximator._base import Approximator
from shapiq.explainer._base import Explainer
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

if TYPE_CHECKING:
    from shapiq.games.imputer.base import Imputer
    from shapiq.utils.custom_types import Model


APPROXIMATOR_CONFIGURATIONS = {
    "regression": {
        "SII": InconsistentKernelSHAPIQ,
        "FSII": RegressionFSII,
        "FBII": RegressionFBII,
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
        "FBII": SHAPIQ,
        "k-SII": SHAPIQ,
        "SV": UnbiasedKernelSHAP,
    },
    "svarm": {
        "SII": SVARMIQ,
        "STII": SVARMIQ,
        "FSII": SVARMIQ,
        "FBII": SVARMIQ,
        "k-SII": SVARMIQ,
        "SV": SVARM,
    },
    "spex": {
        "SII": SPEX,
        "STII": SPEX,
        "FSII": SPEX,
        "FBII": SPEX,
        "k-SII": SPEX,
        "SV": SPEX,
    },
}


class TabularExplainer(Explainer):
    """The tabular explainer as the main interface for the shapiq package.

    The ``TabularExplainer`` class is the main interface for the ``shapiq`` package and tabular
    data. It can be used to explain the predictions of any model by estimating the Shapley
    interaction values.

    Attributes:
        index: Type of Shapley interaction index to use.
        data: A background data to use for the explainer.

    Properties:
        baseline_value: A baseline value of the explainer.

    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        *,
        class_index: int | None = None,
        imputer: Imputer | Literal["marginal", "baseline", "conditional"] = "marginal",
        approximator: Approximator
        | Literal["auto", "spex", "montecarlo", "svarm", "permutation", "regression"] = "auto",
        index: Literal["SII", "k-SII", "STII", "FSII", "FBII", "SV"] = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the TabularExplainer.

        Args:
            model: The model to be explained as a callable function expecting data points as input
                and returning 1-dimensional predictions.

            data: A background dataset to be used for imputation.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            imputer: Either an :class:`~shapiq.games.imputer.Imputer` as implemented in the
                :mod:`~shapiq.games.imputer` module, or a literal string from
                ``["marginal", "baseline", "conditional"]``. Defaults to ``"marginal"``, which
                initializes the default
                :class:`~shapiq.games.imputer.marginal_imputer.MarginalImputer` with its default
                parameters or as provided in ``kwargs``.

            approximator: An :class:`~shapiq.approximator.Approximator` object to use for the
                explainer or a literal string from
                ``["auto", "spex", "montecarlo", "svarm", "permutation"]``. Defaults to ``"auto"``
                which will automatically choose the approximator based on the number of features and
                the desired index.
                    - for index ``"SV"``: :class:`~shapiq.approximator.KernelSHAP`
                    - for index ``"SII"`` or ``"k-SII"``: :class:`~shapiq.approximator.KernelSHAPIQ`
                    - for index ``"FSII"``: :class:`~shapiq.approximator.RegressionFSII`
                    - for index ``"FBII"``: :class:`~shapiq.approximator.RegressionFBII`
                    - for index ``"STII"``: :class:`~shapiq.approximator.SVARMIQ`

            index: The index to explain the model with. Defaults to ``"k-SII"`` which computes the
                k-Shapley Interaction Index. If ``max_order`` is set to 1, this corresponds to the
                Shapley value (``index="SV"``). Options are:
                    - ``"SV"``: Shapley value
                    - ``"k-SII"``: k-Shapley Interaction Index
                    - ``"FSII"``: Faithful Shapley Interaction Index
                    - ``"FBII"``: Faithful Banzhaf Interaction Index (becomes ``BV`` for order 1)
                    - ``"STII"``: Shapley Taylor Interaction Index
                    - ``"SII"``: Shapley Interaction Index

            max_order: The maximum interaction order to be computed. Defaults to ``2``. Set to
                ``1`` for no interactions (single feature importance).

            random_state: The random state to initialize Imputer and Approximator with. Defaults to
                ``None``.

            verbose: Whether to show a progress bar during the computation. Defaults to ``False``.

            **kwargs: Additional keyword-only arguments passed to the imputers implemented in
                :mod:`~shapiq.games.imputer`.
        """
        from shapiq.games.imputer import (
            BaselineImputer,
            ConditionalImputer,
            MarginalImputer,
            TabPFNImputer,
        )

        super().__init__(model, data, class_index)

        # get class for self
        class_name = self.__class__.__name__
        if self._model_type == "tabpfn" and class_name == "TabularExplainer":
            warn(
                "You are using a TabPFN model with the ``shapiq.TabularExplainer`` directly. This "
                "is not recommended as it uses missing value imputation and not contextualization. "
                "Consider using the ``shapiq.TabPFNExplainer`` instead. For more information see "
                "the documentation and the example notebooks.",
                stacklevel=2,
            )

        self._random_state = random_state
        if imputer == "marginal":
            self._imputer = MarginalImputer(
                self.predict,
                self.data,
                random_state=random_state,
                **kwargs,
            )
        elif imputer == "conditional":
            self._imputer = ConditionalImputer(
                self.predict,
                self.data,
                random_state=random_state,
                **kwargs,
            )
        elif imputer == "baseline":
            self._imputer = BaselineImputer(
                self.predict,
                self.data,
                random_state=random_state,
                **kwargs,
            )
        elif isinstance(
            imputer, MarginalImputer | ConditionalImputer | BaselineImputer | TabPFNImputer
        ):
            self._imputer = imputer
        else:
            msg = (
                f"Invalid imputer {imputer}. "
                f'Must be one of ["marginal", "baseline", "conditional"], or a valid Imputer '
                f"object."
            )
            raise ValueError(msg)
        self._n_features: int = self.data.shape[1]
        self._imputer.verbose = verbose  # set the verbose flag for the imputer

        self.index = index
        self._max_order: int = max_order
        self._approximator = self._init_approximator(approximator, self.index, self._max_order)

    def explain_function(
        self,
        x: np.ndarray,
        budget: int,
        random_state: int | None = None,
    ) -> InteractionValues:
        """Explains the model's predictions.

        Args:
            x: The data point to explain as a 2-dimensional array with shape
                (1, n_features).
            budget: The budget to use for the approximation. It indicates how many coalitions are
                sampled, thus high values indicate more accurate approximations, but induce higher
                computational costs.
            random_state: The random state to re-initialize Imputer and Approximator with.
                Defaults to ``None``.

        Returns:
            An object of class :class:`~shapiq.interaction_values.InteractionValues` containing
            the computed interaction values.
        """
        if random_state is not None:
            self._imputer._rng = np.random.default_rng(random_state)  # noqa: SLF001
            self._approximator._rng = np.random.default_rng(random_state)  # noqa: SLF001
            self._approximator._sampler._rng = np.random.default_rng(random_state)  # noqa: SLF001

        # initialize the imputer with the explanation point
        imputer = self._imputer.fit(x)

        # explain
        interaction_values = self._approximator(budget=budget, game=imputer)
        interaction_values.baseline_value = self.baseline_value
        return finalize_computed_interactions(
            interaction_values,
            target_index=self.index,
        )

    @property
    def baseline_value(self) -> float:
        """Returns the baseline value of the explainer."""
        return self._imputer.empty_prediction

    def _init_approximator(
        self,
        approximator: Approximator | str,
        index: str,
        max_order: int,
    ) -> Approximator:
        if isinstance(approximator, Approximator):  # if the approximator is already given
            return approximator

        if approximator == "auto":
            if max_order == 1:
                if index != "SV":
                    warnings.warn(
                        "`max_order=1` but `index != 'SV'`, setting `index = 'SV'`. "
                        "Using the KernelSHAP approximator.",
                        stacklevel=2,
                    )
                    self.index = "SV"
                return KernelSHAP(
                    n=self._n_features,
                    random_state=self._random_state,
                )
            if index == "SV":
                if max_order != 1:
                    warnings.warn(
                        "`index='SV'` but `max_order != 1`, setting `max_order = 1`. "
                        "Using the KernelSHAP approximator.",
                        stacklevel=2,
                    )
                    self._max_order = 1
                return KernelSHAP(
                    n=self._n_features,
                    random_state=self._random_state,
                )
            if index == "FSII":
                return RegressionFSII(
                    n=self._n_features,
                    max_order=max_order,
                    random_state=self._random_state,
                )
            if index == "FBII":
                return RegressionFBII(
                    n=self._n_features,
                    max_order=max_order,
                    random_state=self._random_state,
                )
            if index in ("SII", "k-SII"):
                return KernelSHAPIQ(
                    n=self._n_features,
                    max_order=max_order,
                    random_state=self._random_state,
                    index=index,
                )
            return SVARMIQ(
                n=self._n_features,
                max_order=max_order,
                top_order=False,
                random_state=self._random_state,
                index=index,
            )
        # assume that the approximator is a string
        try:
            approximator_str = approximator.lower()
            approximator = APPROXIMATOR_CONFIGURATIONS[approximator_str][index]
        except KeyError as error:
            msg = (
                f"Invalid approximator `{approximator}` or index `{index}`. "
                f"Valid configuration are described in {APPROXIMATOR_CONFIGURATIONS}."
            )
            raise ValueError(msg) from error
        # initialize the approximator class with params
        return approximator(n=self._n_features, max_order=max_order)
