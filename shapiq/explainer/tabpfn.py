"""This module contains the TabPFNExplainer class, which is a class for explaining the predictions
of a TabPFN model."""

import numpy as np

from ..approximator._base import Approximator
from .tabular import TabularExplainer
from .utils import ModelType, get_predict_function_and_model_type


class TabPFNExplainer(TabularExplainer):
    """The TabPFN explainer as the main interface for the shapiq package.

    The ``TabPFNExplainer`` class is the dedicated interface for the ``shapiq`` package and
    TabPFN[2]_ models such as the ``TabPFNClassifier`` and ``TabPFNRegressor``. The explainer
    does not rely on classical imputation methods and is optimized for TabPFN's in-context learning
    approach. The explanation paradigm for TabPFN is described in Runel et al. (2024)[1]_. In
    essence the explainer is a wrapper around the ``TabularExplainer`` class and uses the same API.

    Args:
        model: Either a TabPFNClassifier or TabPFNRegressor model to be explained.

        data: The background data to use for the explainer as a 2-dimensional array with shape
            ``(n_samples, n_features)``. This data is used to contextualize the model on.

        labels: The labels for the background data as a 1-dimensional array with shape
            ``(n_samples,)``. This data is used to contextualize the model on.

        index: The index to explain the model with. Defaults to ``"k-SII"`` which computes the
            k-Shapley Interaction Index. If ``max_order`` is set to 1, this corresponds to the
            Shapley value (``index="SV"``). Options are:
                - ``"SV"``: Shapley value
                - ``"k-SII"``: k-Shapley Interaction Index
                - ``"FSII"``: Faithful Shapley Interaction Index
                - ``"STII"``: Shapley Taylor Interaction Index
                - ``"SII"``: Shapley Interaction Index (not recommended for XAI since the values do
                    not sum up to the prediction)

        x_test: An optional test data set to compute the model's empty prediction (average
            prediction) on. If no test data and ``empty_prediction`` is set to ``None`` the last
            20% of the background data is used as test data and the remaining 80% as training data
            for contextualization. Defaults to ``None``.

        empty_prediction: Optional value for the model's average prediction on an empty data point
            (all features missing). If provided, overrides parameters in ``x_test``. and skips the
            computation of the empty prediction. Defaults to ``None``.

        class_index: The class index of the model to explain. Defaults to ``None``, which will set
            the class index to ``1`` per default for classification models and is ignored for
            regression models.

        approximator: The approximator to use for calculating the Shapley values or Shapley
            interactions. Can be a string or an instance of an approximator. Defaults to ``"auto"``.

        verbose: Whether to show a progress bar during the computation. Defaults to ``False``.
            Note that verbosity can slow down the computation for large datasets.


    References:
        .. [1] Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., Rügamer, D. (2024). Interpretable Machine Learning for TabPFN. In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2154. Springer, Cham. https://doi.org/10.1007/978-3-031-63797-1_23
        .. [2] Hollmann, N., Müller, S., Purucker, L. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6
    """

    def __init__(
        self,
        *,
        model: ModelType,
        data: np.ndarray,
        labels: np.ndarray,
        index: str = "k-SII",
        max_order: int = 2,
        x_test: np.ndarray | None = None,
        empty_prediction: float | None = None,
        class_index: int | None = None,
        approximator: str | Approximator = "auto",
        verbose: bool = False,
    ):
        from ..games.imputer.tabpfn_imputer import TabPFNImputer

        _predict_function, _ = get_predict_function_and_model_type(model, class_index=class_index)
        model._shapiq_predict_function = _predict_function

        # check that data and labels have the same number of samples
        if data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of samples in `data` and `labels` must be equal (got data.shape= "
                f"{data.shape} and labels.shape={labels.shape})."
            )
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
            return True
        except ImportError:
            return False
