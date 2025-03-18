"""Tabular Explainer class for the shapiq package."""

from warnings import warn

import numpy as np

from ..approximator._base import Approximator
from ..interaction_values import InteractionValues
from ._base import Explainer
from .setup import AVAILABLE_INDICES, setup_approximator
from .validation import set_random_state, validate_budget


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

        super().__init__(
            model,
            data,
            class_index,
            random_state=random_state,
            index=index,
            max_order=max_order,
            verbose=verbose,
        )

        # get class for self
        class_name = self.__class__.__name__
        if self._model_type == "tabpfn" and class_name == "TabularExplainer":
            warn(
                "You are using a TabPFN model with the ``shapiq.TabularExplainer`` directly. This "
                "is not recommended as it uses missing value imputation and not contextualization. "
                "Consider using the ``shapiq.TabPFNExplainer`` instead. For more information see "
                "the documentation and the example notebooks."
            )

        if imputer == "marginal":
            self._imputer = MarginalImputer(
                self.predict, self.data, random_state=self._random_state, **kwargs
            )
        elif imputer == "conditional":
            self._imputer = ConditionalImputer(
                self.predict, self.data, random_state=self._random_state, **kwargs
            )
        elif imputer == "baseline":
            self._imputer = BaselineImputer(
                self.predict, self.data, random_state=self._random_state, **kwargs
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
        self._imputer.verbose = self.verbose  # set the verbose flag for the imputer
        self._approximator = setup_approximator(
            approximator,
            index=self.index,
            max_order=self.max_order,
            n_players=self._n_features,
            random_state=self._random_state,
        )

    def explain_function(
        self,
        x: np.ndarray,
        budget: int | None = None,
        random_state: int | None = None,
        *args,
        **kwargs,
    ) -> InteractionValues:
        """Explains the model's predictions.

        Args:
            x: The data point to explain as a 2-dimensional array with shape
                (1, n_features).
            budget: The budget to use for the approximation. Defaults to `None`, which will
                set the budget to 2**n_features based on the number of features.
            random_state: The random state to re-initialize Imputer and Approximator with. Defaults to ``None``.
        """
        budget = validate_budget(budget, n_players=self._n_features)
        set_random_state(random_state, object_with_rng=self)

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
