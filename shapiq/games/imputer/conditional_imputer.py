"""Implementation of the conditional imputer."""

from typing import Optional

import numpy as np

from shapiq.games.imputer.base import Imputer


class ConditionalImputer(Imputer):
    """A conditional imputer for the shapiq package.

    The conditional imputer is used to impute the missing values of a data point by using the
    conditional distribution estimated with the background data.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        data: The background data to use for the explainer as a two-dimensional array
            with shape (n_samples, n_features).
        x: The explanation point to use the imputer to.
        method: Either 'generative' or TODO 'empirical'.
        sample_size: The number of samples to draw from the conditional background data. Defaults to 10.
        conditional_budget: TODO
        conditional_threshold: TODO
        categorical_features: A list of indices of the categorical features in the background data.
            TODO: not implemented
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`

    Attributes:
        replacement_data: The data to use for imputation. Either samples from the background data
            or the mean/median of the background data.
        empty_prediction: The model's prediction on an empty data point (all features missing).
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        x: np.ndarray = None,
        method="generative",
        sample_size: int = 10,
        conditional_budget: int = 1000,
        conditional_threshold: float = 0.05,
        categorical_features: list[int] = None,
        random_state: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__(model, data, categorical_features, random_state)
        self.method = method
        self.sample_size = sample_size
        self.conditional_budget = conditional_budget
        self.conditional_threshold = conditional_threshold
        self.init_background(data=data)
        if x is not None:
            self.fit(x)
        # set empty value and normalization
        self.empty_prediction: float = self._calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def init_background(self, data: np.ndarray) -> "ConditionalImputer":
        """Intializes the conditional imputer.
        Args:
            data: The background data to use for the imputer. The shape of the array must
                be (n_samples, n_features).

        Returns:
            The initialized imputer.
        """
        import xgboost

        X_tiled = np.tile(data, (self.conditional_budget, 1))
        mask = self._rng.choice(
            [True, False], size=(data.shape[0] * self.conditional_budget, data.shape[1])
        )
        X_masked = X_tiled.copy()
        X_masked[mask] = np.NaN
        tree_embedder = xgboost.XGBRegressor(random_state=self._random_state)
        tree_embedder.fit(X_masked, X_tiled)
        self._data_embedded = tree_embedder.apply(data)
        self._tree_embedder = tree_embedder

    def fit(self, x: np.ndarray[float]) -> "ConditionalImputer":
        """Fits the imputer to the explanation point.

        Args:
            x: The explanation point to use the imputer to.

        Returns:
            The fitted imputer.
        """
        self._x = x
        return self

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray[float]:
        """Computes the value function for all coalitions.

        Args:
            coalitions: A boolean array indicating which features are present (`True`) and which are
                missing (`False`). The shape of the array must be (n_subsets, n_features).

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
               (n_subsets, n_outputs).
        """
        background_data = self._sample_background_data()
        n_coalitions = coalitions.shape[0]
        n_samples = background_data.shape[0]
        x_tiled = np.tile(self._x, (n_coalitions * n_samples, 1))
        background_data_tiled = np.tile(background_data, (n_coalitions, 1))
        coalitions_tiled = np.repeat(coalitions, n_samples, axis=0)
        x_tiled[~coalitions_tiled] = background_data_tiled[~coalitions_tiled]
        predictions = self.predict(x_tiled)
        avg_predictions = predictions.reshape(n_coalitions, -1).mean(axis=1)
        return avg_predictions

    def _sample_background_data(self) -> np.ndarray:
        """Samples background data.

        Returns:
            The sampled replacement values. The shape of the array is (sample_size, n_subsets,
                n_features).
        """
        x_embedded = self._tree_embedder.apply(self._x)
        distances = hamming_distance(self._data_embedded, x_embedded)
        conditional_data = self.data[
            distances <= np.quantile(distances, self.conditional_threshold)
        ]
        if self.sample_size < conditional_data.shape[0]:
            idc = self._rng.choice(conditional_data.shape[0], size=self.sample_size, replace=False)
            background_data = conditional_data[idc, :]
        else:
            background_data = conditional_data
        return background_data

    def _calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        # TODO: perhaps should be self.conditional_data instead of self.data
        empty_predictions = self.predict(self.data)
        empty_prediction = float(np.mean(empty_predictions))
        return empty_prediction


def hamming_distance(X, x):
    """Computes hamming distance between point x (1d) and points in X (2d).
    https://en.wikipedia.org/wiki/Hamming_distance
    """
    x_tiled = np.tile(x, (X.shape[0], 1))
    distances = (X != x_tiled).sum(axis=1)
    return distances
