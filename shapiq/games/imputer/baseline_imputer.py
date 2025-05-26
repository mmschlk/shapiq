"""Implementation of the baseline imputer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .base import Imputer

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


class BaselineImputer(Imputer):
    """The baseline imputer for the shapiq package.

    The baseline imputer is used to impute the missing values of a data point by using predefined
    values (baseline values). If no baseline values are given, the imputer uses the mean (for
    numerical features) or the mode (for categorical features) of the background data.

    Attributes:
        baseline_values: The baseline values to use for imputation.
        empty_prediction: The model's prediction on an empty data point (all features missing).

    Examples:
        >>> model = lambda x: np.sum(x, axis=1)  # some dummy model
        >>> data = np.random.rand(1000, 4)  # some background data
        >>> x_to_impute = np.array([[1, 1, 1, 1]])  # some data point to impute
        >>> imputer = BaselineImputer(model=model, data=data, x=x_to_impute)
        >>> # get the baseline values
        >>> imputer.baseline_values
        array([[0.5, 0.5, 0.5, 0.5]])  # computed from data
        >>> # set new baseline values
        >>> baseline_vector = np.array([0, 0, 0, 0])
        >>> imputer.init_background(baseline_vector)
        >>> imputer.baseline_values
        array([[0, 0, 0, 0]])  # given as input
        >>> # get the model prediction with missing values
        >>> imputer(np.array([[True, False, True, False]]))
        np.array([2.])  # model prediciton with the last baseline value

    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        categorical_features: list[int] | None = None,
        normalize: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initializes the baseline imputer.

        Args:
            model: The model to explain as a callable function expecting a data points as input and
                returning the model's predictions.

            data: The background data to use for the explainer as either a vector of baseline values
                or a two-dimensional array with shape ``(n_samples, n_features)``. If data is a
                matrix, the baseline values are calculated from the data.

            x: The explanation point to use the imputer to.

            categorical_features: A list of indices of the categorical features in the background
                data. If no categorical features are given, all features are assumed to be numerical
                or in string format (where ``np.mean`` fails) features. Defaults to ``None``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            random_state: The random state to use for sampling. Defaults to ``None``.

        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=1,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        # setup attributes
        self.baseline_values: np.ndarray = np.zeros((1, self.n_features))  # will be overwritten
        self.init_background(self.data)

        # set empty value and normalization
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
        for i in range(n_coalitions):
            data[i, ~coalitions[i]] = self.baseline_values[0, ~coalitions[i]]
        return self.predict(data)

    def init_background(self, data: np.ndarray) -> BaselineImputer:
        """Initializes the imputer to the background data.

        Args:
            data: The background data to use for the imputer. Either a vector of baseline values
                of shape ``(n_features,)`` or a matrix of shape ``(n_samples, n_features)``.
                If the data is a matrix, the baseline values are calculated from the data.

        Returns:
            The initialized imputer.

        Examples:
            >>> import numpy as np
            >>> from shapiq.games.imputer import BaselineImputer
            >>> data = np.array([[1, 2, "a"], [2, 3, "a"], [2, 4, "b"]], dtype=object)
            >>> x = np.array([1, 2, 3])
            >>> imputer = BaselineImputer(model=lambda x: np.sum(x, axis=1), data=data, x=x)
            >>> imputer.baseline_values
            array([[1.66, 3, 'a']], dtype=object)  # computed from data
            >>> baseline_vector = np.array([0, 0, 0])
            >>> imputer.init_background(baseline_vector)
            >>> imputer.baseline_values
            array([[0, 0, 0]])  # given as input

        """
        if data.ndim == 1 or data.shape[0] == 1:  # data is a vector -> use as baseline values
            self.baseline_values = data.reshape(1, self.n_features)
            return self
        # data is a matrix -> calculate baseline values as mean or mode
        self.baseline_values = np.zeros((1, self.n_features), dtype=object)
        for feature in range(self.n_features):
            feature_column = data[:, feature]
            if feature in self._cat_features:  # get mode for categorical features
                values, counts = np.unique(feature_column, return_counts=True)
                summarized_feature = values[np.argmax(counts)]
            else:
                try:  # try to use mean for numerical features
                    summarized_feature = np.mean(feature_column)
                except TypeError:  # fallback to mode for potentially string features
                    values, counts = np.unique(feature_column, return_counts=True)
                    summarized_feature = values[np.argmax(counts)]
                    # add feature to categorical features
                    warnings.warn(
                        f"Feature {feature} is not numerical. Adding it to categorical features.",
                        stacklevel=2,
                    )
                    self._cat_features.append(feature)
            self.baseline_values[0, feature] = summarized_feature
        self.calc_empty_prediction()  # reset the empty prediction to the new baseline values
        return self

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.

        """
        empty_predictions = self.predict(self.baseline_values)
        empty_prediction = float(empty_predictions[0])
        self.empty_prediction = empty_prediction
        if self.normalize:  # reset the normalization value
            self.normalization_value = empty_prediction
        return empty_prediction
