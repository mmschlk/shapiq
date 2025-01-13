"""This module contains the TabPFNImputer class, which incoporates the Remove-and-"Retrain" paradigm
of explaining the TabPFN model's predictions."""

from typing import Optional

import numpy as np

from .base import Imputer


class TabPFNImputer(Imputer):

    def __init__(
        self,
        model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: Optional[np.ndarray] = None,
        empty_prediction: Optional[float] = None,
    ):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

        super().__init__(model=model, data=x_test, x=None, sample_size=-1, random_state=None)

        if empty_prediction is None:
            if x_test is None:
                raise ValueError("The empty prediction must be given if no test data is provided.")
            # TODO make sure this works for both classifiers and regressors
            predictions = self.predict(x_test)
            empty_prediction = np.mean(predictions)
        self.empty_prediction = empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """The value function performs the remove-and-"retrain" strategy for TabPFN.

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
            x_explain_coal = self.x[coalition].reshape(1, -1)
            self.model.fit(x_train_coal, self.y_train)
            pred = float(self.predict(x_explain_coal))  # TODO: add class_index
            output[i] = pred
        return output
