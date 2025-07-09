"""Implementation of the conditional imputer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from shapiq.approximator.sampling import CoalitionSampler
from shapiq.utils.modules import check_import_module

from .base import Imputer

if TYPE_CHECKING:
    from typing import Literal

    from shapiq.utils.custom_types import Model


class ConditionalImputer(Imputer):
    """A conditional imputer for the shapiq package.

    The conditional imputer is used to impute the missing values of a data point by using the
    conditional distribution estimated with the background data.

    Attributes:
        empty_prediction: The model's prediction on an empty data point (all features missing).

    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        sample_size: int = 10,
        conditional_budget: int = 128,
        conditional_threshold: float = 0.05,
        normalize: bool = True,
        categorical_features: list[int] | None = None,
        method: Literal["generative"] = "generative",
        random_state: int | None = None,
    ) -> None:
        """Initializes the conditional imputer.

        Args:
            model: The model to explain as a callable function expecting a data points as input and
                returning the model's predictions.

            data: The background data to use for the explainer as a two-dimensional array with shape
                ``(n_samples, n_features)``.

            x: The explanation point to use the imputer on.

            sample_size: The number of samples to draw from the conditional background data for
                imputation. Defaults to ``10``.

            conditional_budget: The number of coallitions to sample per each point in ``data`` for
                training the generative model. Defaults to ``16``.

            conditional_threshold: A quantile threshold defining a neighbourhood of samples to draw
                ``sample_size`` from. A value between ``0.0`` and ``1.0``. Defaults to ``0.05``.

            normalize: A flag to normalize the game values. If ``True`` (default), then the game
                values are normalized and centered to be zero for the empty set of features.
                Defaults to ``True``.

            categorical_features: A list of indices of the categorical features in the background
                data. Currently unused.

            method: The method to use for the conditional imputer. Currently only ``"generative"``
                is implemented. Defaults to ``"generative"``.

            random_state: The random state to use for sampling. Defaults to ``None``.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        if method != "generative":
            msg = "Currently only a generative conditional imputer is implemented."
            raise ValueError(msg)
        self.method = method
        self.conditional_budget = conditional_budget
        self.conditional_threshold = conditional_threshold
        self.init_background(data=data)

        # set empty value and normalization
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def init_background(self, data: np.ndarray) -> ConditionalImputer:
        """Initializes the conditional imputer.

        Args:
            data: The background data to use for the imputer. The shape of the array must
                be (n_samples, n_features).

        Returns:
            The initialized imputer.

        """
        check_import_module("xgboost")
        import xgboost

        n_features = data.shape[1]
        if self.conditional_budget > 2**n_features:
            warnings.warn(
                "`conditional_budget` is higher than `2**n_features`; setting "
                "`conditional_budget = 2**n_features`",
                stacklevel=2,
            )
            self.conditional_budget = 2**n_features
        X_tiled = np.repeat(data, repeats=self.conditional_budget, axis=0)
        coalition_sampler = CoalitionSampler(
            n_players=n_features,
            sampling_weights=np.array([1e-7 for _ in range(n_features + 1)]),
            random_state=self.random_state,
        )
        coalitions_matrix = []
        for _ in range(data.shape[0]):
            coalition_sampler.sample(self.conditional_budget)
            coalitions_matrix.append(coalition_sampler.coalitions_matrix)
        coalitions_matrix = np.concatenate(coalitions_matrix, axis=0)
        X_masked = X_tiled.copy()
        try:
            X_masked[coalitions_matrix] = np.nan  # old numpy version
        except AttributeError:  # interim solution since numpy changed
            X_masked[coalitions_matrix] = np.nan  # new numpy version
        tree_embedder = xgboost.XGBRegressor(random_state=self.random_state)
        tree_embedder.fit(X_masked, X_tiled)
        self._data_embedded = tree_embedder.apply(data)
        self._tree_embedder = tree_embedder
        self._coalition_sampler = coalition_sampler
        return self

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
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
        # insert the better approximate empty prediction for the empty coalitions
        avg_predictions[~np.any(coalitions, axis=1)] = self.empty_prediction
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
            return conditional_data[idc, :]
        return conditional_data

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.

        """
        empty_predictions = self.predict(self.data)
        return float(np.mean(empty_predictions))


def hamming_distance(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute hamming distance between point x (1d) and points in X (2d).

    References:
        - https://en.wikipedia.org/wiki/Hamming_distance
    """
    x_tiled = np.tile(x, (X.shape[0], 1))
    return np.sum(x_tiled != X, axis=1)
