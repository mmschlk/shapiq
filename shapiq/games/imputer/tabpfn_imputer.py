"""TabPFNImputer module.

This module contains the TabPFNImputer class, which incorporates the Remove-and-Contextualize
paradigm of explaining the TabPFN model's predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Imputer

if TYPE_CHECKING:
    from collections.abc import Callable

    from tabpfn import TabPFNClassifier, TabPFNRegressor

    from shapiq.typing import Model


class TabPFNImputer(Imputer):
    """An Imputer for TabPFN using the Remove-and-Contextualize paradigm.

    The remove-and-contextualize paradigm is a strategy to explain the predictions of a TabPFN [2]_
    model which uses in-context learning for prediction. Instead of imputing missing features, the
    TabPFNImputer removes feature columns missing in a coalition from training data and re-"trains"
    re-contextualizes the model with the remaining features. The model is then used to predict the
    data point which is also missing the features. This pardigm is described in Rundel et al.
    (2024) [1]_.

    Attributes:
        x_train: The training data to contextualize the model on.
        y_train: The training labels to contextualize the model on.
        empty_prediction: The model's average prediction on an empty data point.

    References:
        .. [1] Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., Rügamer, D. (2024). Interpretable Machine Learning for TabPFN. In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2154. Springer, Cham. https://doi.org/10.1007/978-3-031-63797-1_23
        .. [2] Hollmann, N., Müller, S., Purucker, L. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319-326 (2025). https://doi.org/10.1038/s41586-024-08328-6

    """

    def __init__(
        self,
        model: TabPFNClassifier | TabPFNRegressor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_test: np.ndarray | None = None,
        empty_prediction: float | None = None,
        verbose: bool = False,
        predict_function: Callable[[Model, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """An Imputer for TabPFN using the Remove-and-Contextualize paradigm.

        Args:
            model: The model to be explained as a callable function expecting data points as input
                and returning 1-dimensional predictions.

            x_train: The training data to "train" the model on. Note that the model is not actually
                trained but the correct train data with the correct features per coalition are put
                into TabPFN's context.

            y_train: The training labels to "train" the model on. Note that the model is not trained
                but the correct train data and labels are put into TabPFN's context.

            x_test: The test data to evaluate the model's average (empty) prediction on. If no test
                data is provided, the empty prediction must be given. Defaults to ``None``.

            empty_prediction: The model's average prediction on an empty data point (all features
                missing). This can be computed by averaging the model's predictions on the test
                data.

            verbose: A flag to enable verbose output. Defaults to ``False``.

            predict_function: A function to use for prediction. If the model is not instantiated via
                a ``shapiq.Explainer`` object, this function must be provided. The function must
                accept the model and the data point as input and return the model's predictions. If
                the model is instantiated via a ``shapiq.Explainer`` object, this function is
                automatically set to the model's prediction function. Defaults to ``None``.

        """
        self.x_train = x_train
        self.y_train = y_train

        if not hasattr(model, "_shapiq_predict_function"):
            if predict_function is None:
                msg = (
                    f"If the Imputer is not instantiated via a ``shapiq.Explainer`` object, you"
                    f" must provide a ``predict_function`` (received"
                    f" predict_function={predict_function})."
                )
                raise ValueError(msg)
            model._shapiq_predict_function = predict_function  # noqa: SLF001

        if x_test is None and empty_prediction is None:
            msg = "The empty prediction must be given if no test data is provided"
            raise ValueError(msg)

        super().__init__(
            model=model,
            data=x_test,
            x=None,
            sample_size=None,
            random_state=None,
            verbose=verbose,
        )

        if empty_prediction is None:
            self.model.fit(x_train, y_train)  # contextualize the model on the training data
            predictions = self.predict(x_test)
            empty_prediction = np.mean(predictions)
        self.empty_prediction = empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """The value function performs the remove-and-contextualize strategy for TabPFN.

        The value function removes absent features from a coalition by "training" the model again
        on the subset of features. The model is then used to predict the data point with the
        missing features.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which
                are missing (``False``). The shape of the array must be ``(n_subsets, n_players)``.

        Returns:
            The model's predictions on the restricted data points. The shape of the array is
                ``(n_subsets,)``.

        """
        output = np.zeros(len(coalitions), dtype=float)
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                output[i] = self.empty_prediction
                continue
            x_train_coal = self.x_train[:, coalition]
            x_explain_coal = self.x[:, coalition]
            self.model.fit(x_train_coal, self.y_train)
            pred = float(self.predict(x_explain_coal))
            output[i] = pred
        # refit the model on the full training data to ensure it is in a consistent state
        self.model.fit(self.x_train, self.y_train)
        return output
