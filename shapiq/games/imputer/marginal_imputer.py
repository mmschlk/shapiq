"""Implementation of the marginal imputer."""

import warnings
from typing import Optional

import numpy as np

from shapiq.games.imputer.base import Imputer


class MarginalImputer(Imputer):
    """The marginal imputer for the shapiq package.

    The marginal imputer is used to impute the missing values of a data point by using the
    marginal distribution of the background data.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        data: The background data to use for the explainer as a two-dimensional array
            with shape ``(n_samples, n_features)``.
        x: The explanation point to use the imputer to.
        sample_size: The number of samples to draw from the background data. Only used if
            ``sample_replacements`` is ``True``. Increasing this value will linearly increase the
            runtime of the explainer. Defaults to ``100``.
        categorical_features: A list of indices of the categorical features in the background data.
            If no categorical features are given, all features are assumed to be numerical or in
            string format (where ``np.mean`` fails) features. Defaults to ``None``.
        normalize: A flag to normalize the game values. If ``True``, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to ``True``.
        random_state: The random state to use for sampling. Defaults to ``None``.

    Attributes:
        replacement_data: The data to use for imputation. Either samples from the background data
            or the mean / median of the background data.
        empty_prediction: The model's prediction on an empty data point (all features missing).
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        x: Optional[np.ndarray] = None,
        sample_replacements: bool = True,
        sample_size: int = 100,
        categorical_features: list[int] = None,
        normalize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if not sample_replacements:
            warnings.warn(
                "The 'sample_replacements' argument is deprecated and will be removed in the next "
                "release. The marginal imputer now always samples from the background data.",
                DeprecationWarning,
            )
        super().__init__(model, data, sample_size, categorical_features, random_state)

        # setup attributes
        self.replacement_data: np.ndarray = np.zeros((1, self._n_features))  # will be overwritten
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
        # sampling from background returning array of shape (sample_size, n_subsets, n_features)
        replacement_data = self._sample_replacement_values(coalitions)
        outputs = np.zeros((self.sample_size, n_coalitions))
        for i in range(self.sample_size):
            replacements = replacement_data[i].reshape(n_coalitions, self._n_features)
            data[~coalitions] = replacements[~coalitions]
            outputs[i] = self.predict(data)
        outputs = np.mean(outputs, axis=0)  # average over the samples
        return outputs

    def init_background(self, data: np.ndarray) -> "MarginalImputer":
        """Initializes the imputer to the background data.

        Args:
            data: The background data to use for the imputer. The shape of the array must
                be ``(n_samples, n_features)``.

        Returns:
            The initialized imputer.
        """
        self.replacement_data = data
        return self

    def fit(self, x: np.ndarray) -> "MarginalImputer":
        """Fits the imputer to the explanation point.

        Args:
            x: The explanation point to use the imputer to.

        Returns:
            The fitted imputer.
        """
        self._x = x
        return self

    def _sample_replacement_values(self, coalitions: np.ndarray) -> np.ndarray:
        """Samples replacement values from the background data.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which are
                missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The sampled replacement values. The shape of the array is ``(sample_size, n_subsets,
                n_features)``.
        """
        n_coalitions = coalitions.shape[0]
        replacement_data = np.zeros(
            (self.sample_size, n_coalitions, self._n_features), dtype=object
        )
        for feature in range(self._n_features):
            sampled_feature_values = self._rng.choice(
                self.replacement_data[:, feature],
                size=(self.sample_size, n_coalitions),
                replace=True,
            )
            replacement_data[:, :, feature] = sampled_feature_values
        return replacement_data

    def _calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        empty_predictions = self.predict(self.replacement_data)
        empty_prediction = np.mean(empty_predictions)
        return empty_prediction
