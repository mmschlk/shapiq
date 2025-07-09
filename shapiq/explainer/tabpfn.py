"""Implementation of TabPFNExplainer class.

The TabPFNExplainer is a class for explaining the predictions of a TabPFN model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .tabular import TabularExplainer
from .utils import get_predict_function_and_model_type

if TYPE_CHECKING:
    from typing import Literal

    from shapiq.approximator.base import Approximator
    from shapiq.utils.custom_types import Model

    from .custom_types import ExplainerIndices


class TabPFNExplainer(TabularExplainer):
    """The TabPFN explainer as the main interface for the shapiq package.

    The ``TabPFNExplainer`` class is the dedicated interface for the ``shapiq`` package and
    TabPFN :footcite:t:`Hollmann.2025` models such as the ``TabPFNClassifier`` and
    ``TabPFNRegressor``. The explainer does not rely on classical imputation methods and is
    optimized for TabPFN's in-context learning approach. The explanation paradigm for TabPFN is
    described in :footcite:t:`Rundel.2024`. In essence the explainer is a wrapper around the
    :class:~`shapiq.explainer.tabular.TabularExplainer` class and uses the same API.

    References:
        .. footbibliography::
    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        index: ExplainerIndices = "k-SII",
        max_order: int = 2,
        x_test: np.ndarray | None = None,
        empty_prediction: float | None = None,
        class_index: int | None = None,
        approximator: Approximator
        | Literal["auto", "spex", "montecarlo", "svarm", "permutation", "regression"] = "auto",
        verbose: bool = False,
    ) -> None:
        """Initialize the TabPFNExplainer.

        Args:
            model: Either a TabPFNClassifier or TabPFNRegressor model to be explained.

            data: The background data to use for the explainer as a 2-dimensional array with shape
                ``(n_samples, n_features)``. This data is used to contextualize the model on.

            labels: The labels for the background data as a 1-dimensional array with shape
                ``(n_samples,)``. This data is used to contextualize the model on.

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

            x_test: An optional test data set to compute the model's empty prediction (average
                prediction) on. If no test data and ``empty_prediction`` is set to ``None`` the last
                20% of the background data is used as test data and the remaining 80% as training
                data for contextualization. Defaults to ``None``.

            empty_prediction: Optional value for the model's average prediction on an empty data
                point (all features missing). If provided, overrides parameters in ``x_test``. and
                skips the computation of the empty prediction. Defaults to ``None``.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            verbose: Whether to show a progress bar during the computation. Defaults to ``False``.
                Note that verbosity can slow down the computation for large datasets.

        """
        from shapiq.games.imputer.tabpfn_imputer import TabPFNImputer

        _predict_function, _ = get_predict_function_and_model_type(model, class_index=class_index)
        model._shapiq_predict_function = _predict_function  # noqa: SLF001

        # check that data and labels have the same number of samples
        if data.shape[0] != labels.shape[0]:
            msg = (
                f"The number of samples in `data` and `labels` must be equal (got data.shape= "
                f"{data.shape} and labels.shape={labels.shape})."
            )
            raise ValueError(msg)
        n_samples = data.shape[0]
        x_train = data
        y_train = labels

        if x_test is None and empty_prediction is None:
            sections = [int(0.8 * n_samples)]
            x_train, x_test = np.split(data, sections)
            y_train, _ = np.split(labels, sections)

        if x_test is None:
            x_test = x_train  # is not used in the TabPFNImputer if empty_prediction is set

        imputer = TabPFNImputer(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            empty_prediction=empty_prediction,
            verbose=verbose,
        )

        super().__init__(
            model,
            data=x_test,
            imputer=imputer,
            class_index=class_index,
            approximator=approximator,
            index=index,
            max_order=max_order,
        )

    @property
    def is_available(self) -> bool:
        """Check if the TabPFN package is available."""
        import importlib

        try:
            importlib.import_module("tabpfn")
        except ImportError:
            return False
        return True
