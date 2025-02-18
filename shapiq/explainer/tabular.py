"""Tabular Explainer class for the shapiq package."""

import warnings
from warnings import warn

import numpy as np

from ..approximator import (
    SHAPIQ,
    SVARM,
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
from ..approximator._base import Approximator
from ..explainer._base import Explainer
from ..interaction_values import InteractionValues

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
    "svarm": {
        "SII": SVARMIQ,
        "STII": SVARMIQ,
        "FSII": SVARMIQ,
        "k-SII": SVARMIQ,
        "SV": SVARM,
    },
}

AVAILABLE_INDICES = {"SII", "k-SII", "STII", "FSII", "SV"}


class TabularExplainer(Explainer):
    """The tabular explainer as the main interface for the shapiq package.

    The ``TabularExplainer`` class is the main interface for the ``shapiq`` package and tabular
    data. It can be used to explain the predictions of any model by estimating the Shapley
    interaction values.

    Args:
        model: The model to be explained as a callable function expecting data points as input and
            returning 1-dimensional predictions.

        data: A background dataset to be used for imputation.

        class_index: The class index of the model to explain. Defaults to ``None``, which will set
            the class index to ``1`` per default for classification models and is ignored for
            regression models.

        imputer: Either an object of class Imputer or a string from ``["marginal", "conditional"]``.
            Defaults to ``"marginal"``, which innitializes the default MarginalImputer.

        approximator: An approximator object to use for the explainer. Defaults to ``"auto"``
            which will automatically choose the approximator based on the number of features and
            the desired index.
                - for index ``"SV"``: :class:`~shapiq.approximator.KernelSHAP`
                - for index ``"SII"`` or ``"k-SII"``: :class:`~shapiq.approximator.KernelSHAPIQ`
                - for index ``"FSII"``: :class:`~shapiq.approximator.RegressionFSII`
                - for index ``"STII"``: :class:`~shapiq.approximator.SVARMIQ`

        index: The index to explain the model with. Defaults to ``"k-SII"`` which computes the
            k-Shapley Interaction Index. If ``max_order`` is set to 1, this corresponds to the
            Shapley value (``index="SV"``). Options are:
                - ``"SV"``: Shapley value
                - ``"k-SII"``: k-Shapley Interaction Index
                - ``"FSII"``: Faithful Shapley Interaction Index
                - ``"STII"``: Shapley Taylor Interaction Index
                - ``"SII"``: Shapley Interaction Index (not recommended for XAI since the values do
                    not sum up to the prediction)

        max_order: The maximum interaction order to be computed. Defaults to ``2``. Set to ``1`` for
            no interactions (single feature importance).

        random_state: The random state to initialize Imputer and Approximator with. Defaults to
            ``None``.

        **kwargs: Additional keyword-only arguments passed to the imputer.

    Attributes:
        index: Type of Shapley interaction index to use.
        data: A background data to use for the explainer.

    Properties:
        baseline_value: A baseline value of the explainer.
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        class_index: int | None = None,
        imputer="marginal",
        approximator: str | Approximator = "auto",
        index: str = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        from shapiq.games.imputer import (
            BaselineImputer,
            ConditionalImputer,
            MarginalImputer,
            TabPFNImputer,
        )

        if index not in AVAILABLE_INDICES:
            raise ValueError(f"Invalid index `{index}`. " f"Valid indices are {AVAILABLE_INDICES}.")

        super().__init__(model, data, class_index)

        # get class for self
        class_name = self.__class__.__name__
        if self._model_type == "tabpfn" and class_name == "TabularExplainer":
            warn(
                "You are using a TabPFN model with the ``shapiq.TabularExplainer`` directly. This "
                "is not recommended as it uses missing value imputation and not contextualization. "
                "Consider using the ``shapiq.TabPFNExplainer`` instead. For more information see "
                "the documentation and the example notebooks."
            )

        self._random_state = random_state
        if imputer == "marginal":
            self._imputer = MarginalImputer(
                self.predict, self.data, random_state=random_state, **kwargs
            )
        elif imputer == "conditional":
            self._imputer = ConditionalImputer(
                self.predict, self.data, random_state=random_state, **kwargs
            )
        elif imputer == "baseline":
            self._imputer = BaselineImputer(
                self.predict, self.data, random_state=random_state, **kwargs
            )
        elif (
            isinstance(imputer, MarginalImputer)
            or isinstance(imputer, ConditionalImputer)
            or isinstance(imputer, BaselineImputer)
            or isinstance(imputer, TabPFNImputer)
        ):
            self._imputer = imputer
        else:
            raise ValueError(
                f"Invalid imputer {imputer}. "
                f'Must be one of ["marginal", "baseline", "conditional"], or a valid Imputer '
                f"object."
            )
        self._n_features: int = self.data.shape[1]
        self._imputer.verbose = verbose  # set the verbose flag for the imputer

        self.index = index
        self._max_order: int = max_order
        self._approximator = self._init_approximator(approximator, self.index, self._max_order)

    def explain_function(
        self, x: np.ndarray, budget: int | None = None, random_state: int | None = None
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
        interaction_values = self._approximator(budget=budget, game=imputer)
        interaction_values.baseline_value = self.baseline_value

        return interaction_values

    @property
    def baseline_value(self) -> float:
        """Returns the baseline value of the explainer."""
        return self._imputer.empty_prediction

    def _init_approximator(
        self, approximator: Approximator | str, index: str, max_order: int
    ) -> Approximator:

        if isinstance(approximator, Approximator):  # if the approximator is already given
            return approximator

        if approximator == "auto":
            if max_order == 1:
                if index != "SV":
                    warnings.warn(
                        "`max_order=1` but `index != 'SV'`, setting `index = 'SV'`. "
                        "Using the KernelSHAP approximator."
                    )
                    self.index = "SV"
                return KernelSHAP(
                    n=self._n_features,
                    random_state=self._random_state,
                )
            elif index == "SV":
                if max_order != 1:
                    warnings.warn(
                        "`index='SV'` but `max_order != 1`, setting `max_order = 1`. "
                        "Using the KernelSHAP approximator."
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
