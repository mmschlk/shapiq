"""This module contains the marginal imputer for the shapiq package."""

from typing import Callable, Optional

import numpy as np
from explainer.imputer._base import Imputer


class MarginalImputer(Imputer):
    """The marginal imputer for the shapiq package.

    The marginal imputer is used to impute the missing values of a data point by using the
    marginal distribution of the background data.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        background_data: The background data to use for the explainer as a two-dimensional array
            with shape (n_samples, n_features).
        sample_replacements: Whether to sample replacements from the background data or to use the
            mean (for numerical features) or the median (for categorical features) of the background
            data. Defaults to `False`.
        sample_size: The number of samples to draw from the background data. Only used if
            `sample_replacements` is `True`. Increasing this value will linearly increase the
            runtime of the explainer. Defaults to `1`.
        categorical_features: A list of indices of the categorical features in the background data.
            If no categorical features are given, all features are assumed to be numerical or in
            string format (where `np.mean` fails) features. Defaults to `None`.

    Attributes:
        replacement_data: The data to use for imputation. Either samples from the background data
            or the mean/median of the background data.
        empty_prediction: The model's prediction on an empty data point (all features missing).
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        x_explain: Optional[np.ndarray] = None,
        sample_replacements: bool = False,
        sample_size: int = 1,
        categorical_features: list[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(model, background_data, categorical_features, random_state)
        self._sample_replacements = sample_replacements
        self._sample_size: int = sample_size
        self.replacement_data: np.ndarray = np.zeros((1, self._n_features))  # will be overwritten
        self.init_background(self._background_data)
        self._x_explain: np.ndarray = np.zeros((1, self._n_features))  # will be overwritten @ fit
        if x_explain is not None:
            self.fit(x_explain)
        self.empty_prediction: float = self._calc_empty_prediction()

    def __call__(self, subsets: np.ndarray[bool]) -> np.ndarray[float]:
        """Imputes the missing values of a data point and calls the model.

        Args:
            subsets: A boolean array indicating which features are present (`True`) and which are
                missing (`False`). The shape of the array must be (n_subsets, n_features).

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
               (n_subsets, n_outputs).
        """
        n_subsets = subsets.shape[0]
        data = np.tile(np.copy(self._x_explain), (n_subsets, 1))
        if not self._sample_replacements:
            replacement_data = np.tile(self.replacement_data, (n_subsets, 1))
            data[~subsets] = replacement_data[~subsets]
            outputs = self._model(data)
        else:
            # sampling from background returning array of shape (sample_size, n_subsets, n_features)
            replacement_data = self._sample_replacement_values(subsets)
            outputs = np.zeros((self._sample_size, n_subsets))
            for i in range(self._sample_size):
                replacements = replacement_data[i].reshape(n_subsets, self._n_features)
                data[~subsets] = replacements[~subsets]
                outputs[i] = self._model(data)
            outputs = np.mean(outputs, axis=0)  # average over the samples
        outputs -= self.empty_prediction
        return outputs

    def init_background(self, x_background: np.ndarray) -> "MarginalImputer":
        """Initializes the imputer to the background data.

        Args:
            x_background: The background data to use for the imputer. The shape of the array must
                be (n_samples, n_features).

        Returns:
            The initialized imputer.
        """
        if self._sample_replacements:
            self.replacement_data = x_background
        else:
            self.replacement_data = np.zeros((1, self._n_features), dtype=object)
            for feature in range(self._n_features):
                feature_column = x_background[:, feature]
                if feature in self._cat_features:
                    # get mode for categorical features
                    counts = np.unique(feature_column, return_counts=True)
                    summarized_feature = counts[0][np.argmax(counts[1])]
                else:
                    try:  # try to use mean for numerical features
                        summarized_feature = np.mean(feature_column)
                    except TypeError:  # fallback to mode for string features
                        counts = np.unique(feature_column, return_counts=True)
                        summarized_feature = counts[0][np.argmax(counts[1])]
                self.replacement_data[:, feature] = summarized_feature
        return self

    def fit(self, x_explain: np.ndarray[float]) -> "MarginalImputer":
        """Fits the imputer to the explanation point.

        Args:
            x_explain: The explanation point to use the imputer to.

        Returns:
            The fitted imputer.
        """
        self._x_explain = x_explain
        return self

    def _sample_replacement_values(self, subsets: np.ndarray[bool]) -> np.ndarray:
        """Samples replacement values from the background data.

        Args:
            subsets: A boolean array indicating which features are present (`True`) and which are
                missing (`False`). The shape of the array must be (n_subsets, n_features).

        Returns:
            The sampled replacement values. The shape of the array is (sample_size, n_subsets,
                n_features).
        """
        n_subsets = subsets.shape[0]
        replacement_data = np.zeros((self._sample_size, n_subsets, self._n_features), dtype=object)
        for feature in range(self._n_features):
            sampled_feature_values = self._rng.choice(
                self.replacement_data[:, feature], size=(self._sample_size, n_subsets), replace=True
            )
            replacement_data[:, :, feature] = sampled_feature_values
        return replacement_data

    def _calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        if self._sample_replacements:
            shuffled_background = self._rng.permutation(self._background_data)
            empty_predictions = self._model(shuffled_background)
            empty_prediction = float(np.mean(empty_predictions))
            return empty_prediction
        empty_prediction = self._model(self.replacement_data)
        empty_prediction = float(empty_prediction)
        return empty_prediction
