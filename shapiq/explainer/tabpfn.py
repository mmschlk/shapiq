"""This module contains the TabPFNExplainer class, which is a class for explaining the predictions
of a TabPFN model."""

from typing import Optional, Union

import numpy as np

from ..approximator._base import Approximator
from .tabular import TabularExplainer


class TabPFNExplainer(TabularExplainer):
    """The TabPFN explainer as the main interface for the shapiq package.

    The ``TabPFNExplainer`` class is the dedicated interface for the ``shapiq`` package and
    TabPFN[2]_ models such as the ``TabPFNClassifier`` and ``TabPFNRegressor``. The explainer
    does not rely on classical imputation methods and is optimized for TabPFN's in-context learning
    approach. The explanation paradigm for TabPFN is described in Runel et al. (2024)[1]_. In
    essence the explainer is a wrapper around the ``TabularExplainer`` class and uses the same API.

    Args:
        model: Either a TabPFNClassifier or TabPFNRegressor model to be explained.

    References:
        .. [1] Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., Rügamer, D. (2024). Interpretable Machine Learning for TabPFN. In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2154. Springer, Cham. https://doi.org/10.1007/978-3-031-63797-1_23
        .. [2] Hollmann, N., Müller, S., Purucker, L. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6
    """

    def __init__(
        self,
        *,
        model,
        x_train,
        y_train,
        x_test: Optional[np.ndarray] = None,
        empty_prediction: Optional[float] = None,
        class_index: Optional[int] = None,
        approximator: Union[str, Approximator] = "auto",
        index: str = "k-SII",
        max_order: int = 2,
    ):
        from ..games.imputer.tabpfn_imputer import TabPFNImputer

        imputer = TabPFNImputer(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            empty_prediction=empty_prediction,
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
