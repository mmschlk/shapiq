"""Implementation of the baseline imputer."""

from typing import Optional

import numpy as np

from shapiq.games.imputer.base import Imputer


class BaselineImputer(Imputer):
    """The baseline imputer for the shapiq package.

    The baseline imputer is used to impute the missing values of a data point by using predefined
    values (baseline values). If no baseline values are given, the imputer uses the mean (for
    numerical features) or the mode (for categorical features) of the background data.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        data: The background data to use for the explainer as either a vector of baseline values
            or a two-dimensional array with shape ``(n_samples, n_features)``. If data is a matrix,
            the baseline values are calculated from the data.
        x: The explanation point to use the imputer to.
        categorical_features: A list of indices of the categorical features in the background data.
            If no categorical features are given, all features are assumed to be numerical or in
            string format (where ``np.mean`` fails) features. Defaults to ``None``.
        normalize: A flag to normalize the game values. If ``True``, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to ``True``.
        random_state: The random state to use for sampling. Defaults to ``None``.

    Attributes:
        baseline_vector: The baseline values to use for imputation.
        empty_prediction: The model's prediction on an empty data point (all features missing).
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        x: Optional[np.ndarray] = None,
        categorical_features: list[int] = None,
        normalize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(model, data, 1, categorical_features, random_state)

        # setup attributes
        self.baseline_vector: np.ndarray = np.zeros((1, self._n_features))  # will be overwritten
        self.init_background(self.data)
        self._x: np.ndarray = np.zeros((1, self._n_features))  # will be overwritten @ fit
        if x is not None:
            self.fit(x)

        # set empty value and normalization
        self.empty_prediction: float = self._calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Imputes the missing values of a data point and calls the model.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which are
                missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
               ``(n_subsets, n_outputs)``.
        """
        n_coalitions = coalitions.shape[0]
        data = np.tile(np.copy(self._x), (n_coalitions, 1))
        baseline_vector = np.tile(self.baseline_vector, (n_coalitions, 1))
        data[~coalitions] = baseline_vector[~coalitions]
        outputs = self.predict(data)
        return outputs

    def init_background(self, data: np.ndarray) -> "BaselineImputer":
        """Initializes the imputer to the background data.

        Args:
            data: The background data to use for the imputer. Either a vector of baseline values
                of shape ``(1, n_features)`` or a matrix of shape ``(n_samples, n_features)``.
                If the data is a matrix, the baseline values are calculated from the data.

        Returns:
            The initialized imputer.
        """
        if data.ndim == 1:  # if data is a vector of baseline values
            self.baseline_vector = data.reshape(1, -1)
            return self
        # data is a matrix -> calculate baseline values as mean or mode
        self.baseline_vector = np.zeros((1, self._n_features), dtype=object)
        for feature in range(self._n_features):
            feature_column = data[:, feature]
            if feature in self._cat_features:  # get mode for categorical features
                counts = np.unique(feature_column, return_counts=True)
                summarized_feature = counts[0][np.argmax(counts[1])]
            else:
                try:  # try to use mean for numerical features
                    summarized_feature = np.mean(feature_column)
                except TypeError:  # fallback to mode for potentially string features
                    counts = np.unique(feature_column, return_counts=True)
                    summarized_feature = counts[0][np.argmax(counts[1])]
            self.baseline_vector[0, feature] = summarized_feature
        return self

    def fit(self, x: np.ndarray) -> "BaselineImputer":
        """Fits the imputer to the explanation point.

        Args:
            x: The explanation point to use the imputer on.

        Returns:
            The fitted imputer.
        """
        self._x = x
        return self

    def _calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        empty_predictions = self.predict(self.baseline_vector)
        empty_prediction = empty_predictions[0]
        return float(empty_prediction)
