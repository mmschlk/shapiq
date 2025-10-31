"""Implements the Gaussian-based approach for imputation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.random import default_rng

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from shapiq.game import Game

from .base import Imputer
from .gaussian_imputer_exceptions import CategoricalFeatureError

# We disallow columns with <= 2 unique values, since they are likely either:
# - Binary features
# - One-hot encoded features (which would have at most 2 values per encoded column)
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 2


class GaussianImputer(Imputer):
    r"""Implements the Gaussian-based approach for imputation according to [Aas21]_.

    This approach assumes that the features of the background data form a multivariate Gaussian distribution.
    The missing values are imputed by drawing Monte Carlo samples from the conditional distribution given the values of the features present in a coalition.

    Note that only continuous features are supported, meaning that this imputer can't be used for datasets containing
    categorical or binary features.
    """

    def __init__(
        self,
        model: object | Game | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        sample_size: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the class.

        Args:
            model: The model to explain as a callable function expecting a data points as input and
                returning the model's predictions.

            data: The background data to use for the explainer as a two-dimensional array with shape
                ``(n_samples, n_features)``.

            x: The explanation point as a ``np.ndarray`` of shape ``(1, n_features)`` or
                ``(n_features,)``.

            sample_size: The number of Monte Carlo samples to draw from the conditional background
                data for imputation.

            random_state: An optional random seed for reproducibility.

            verbose: A flag to enable verbose imputation, which will print a progress bar for model
                evaluation. Note that this can slow down the imputation process.

        Raises:
            CategoricalFeatureError: If the background data contains any categorical features.
        """
        if data.shape[0] == 0:
            msg = "Background data must not be empty"
            raise ValueError(msg)

        Imputer.__init__(
            self,
            model=model,
            data=data,
            x=x,
            categorical_features=[],
            sample_size=sample_size,
            random_state=random_state,
            verbose=verbose,
        )

        self._check_categorical_features()

        self._mean_per_feature: npt.NDArray[np.floating] | None = None
        self._cov_mat: npt.NDArray[np.floating] | None = None

    def _check_categorical_features(self) -> None:
        """Check if any features are categorical variables.

        Raises:
            CategoricalFeatureError: If any categorical features are detected.
        """
        categorical_indices: list[int] = []

        for i, feature_values in enumerate(self.data.T):
            if any(isinstance(v, str) for v in feature_values):
                categorical_indices.append(i)
                continue
            unique_values = len(np.unique(feature_values))
            if unique_values <= MAX_UNIQUE_VALUES_FOR_CATEGORICAL:
                categorical_indices.append(i)

        if len(categorical_indices) > 0:
            raise CategoricalFeatureError(categorical_indices)

    @property
    def mean_per_feature(self) -> npt.NDArray[np.floating]:
        """The mean value for each feature.

        This proprety is only computed once and then cached.
        """
        if self._mean_per_feature is None:
            self._mean_per_feature = cast("npt.NDArray[np.floating]", np.mean(self.data, axis=0))
        return self._mean_per_feature

    @property
    def cov_mat(self) -> npt.NDArray[np.floating]:
        """The covariance matrix of the features.

        This proprety is only computed once and then cached.
        """
        if self._cov_mat is None:
            self._cov_mat = self._ensure_positive_definite(np.cov(self.data.T))
        return self._cov_mat

    def _ensure_positive_definite(
        self,
        cov_mat: npt.NDArray[np.floating],
        min_allowed_eigen_value: float = 1e-06,
    ) -> npt.NDArray[np.floating]:
        """Ensure covariance matrix is positive definite by correcting eigenvalues if necessary.

        Args:
            cov_mat: The input covariance matrix.
            min_allowed_eigen_value: The minimum allowed eigenvalue. Defaults to ``1e-06``.

        Returns:
            The positive definite covariance matrix.
        """
        eigen_values = np.linalg.eigvalsh(cov_mat)

        if np.any(eigen_values <= min_allowed_eigen_value):
            # Add regularization to make the matrix positive definite
            min_eigen_value = np.min(eigen_values)
            cov_mat += (min_allowed_eigen_value - min_eigen_value) * np.eye(cov_mat.shape[0])

        return cov_mat

    def _sample_monte_carlo(
        self,
        x: npt.NDArray[np.floating],
        coalitions: npt.NDArray[np.bool],
    ) -> npt.NDArray[np.floating]:
        """Generate Gaussian Monte Carlo samples for the features missing in the given coalitions.

        Args:
            coalitions: The coalitions for which to impute values as a boolean array of shape ``(n_coalitions, n_features)``.
            x: The explanation point to use the imputer on.

        Returns:
            Random samples for the missing features of each coalition as an array of shape ``(n_coalitions, sample_size, n_features)``.
                The columns corresponding to known features are filled with the value of ``x`` for that feature.
        """
        x_explain = x.flatten()
        n_coalitions, n_features = coalitions.shape
        rng = default_rng(self.random_state)

        samples_all_coalitions = np.zeros((n_coalitions, self.sample_size, n_features))

        for i, coalition in enumerate(coalitions):
            known_indices = np.where(coalition)[0]
            unknown_indices = np.where(~coalition)[0]

            if len(known_indices) == 0:
                # No conditioning on known features, therefore sample from original data distribution
                Z = rng.standard_normal((self.sample_size, len(unknown_indices)))
                samples = Z @ np.linalg.cholesky(self.cov_mat).T + self.mean_per_feature
            elif len(unknown_indices) == 0:
                samples = np.tile(x_explain, (self.sample_size, 1))
            else:
                x_S_star = x_explain[known_indices]

                mu_S_known = self.mean_per_feature[known_indices]
                mu_S_unknown = self.mean_per_feature[unknown_indices]

                cov_S_known_known = self.cov_mat[np.ix_(known_indices, known_indices)]
                cov_S_known_unknown = self.cov_mat[np.ix_(known_indices, unknown_indices)]
                cov_S_unknown_known = self.cov_mat[np.ix_(unknown_indices, known_indices)]
                cov_S_unknown_unknown = self.cov_mat[np.ix_(unknown_indices, unknown_indices)]

                cov_S_known_known_inv = np.linalg.inv(cov_S_known_known)

                cond_mean = mu_S_unknown + (cov_S_unknown_known @ cov_S_known_known_inv) @ (
                    x_S_star - mu_S_known
                )
                cond_cov = (
                    cov_S_unknown_unknown
                    - (cov_S_unknown_known @ cov_S_known_known_inv) @ cov_S_known_unknown
                )
                # for sampling from multivariate normal distribution with Cholesky we need to make sure that
                # cond_cov is symmetric (regardless - Covariances should always be symmetric: Cov(X,Y) = Cov(Y,X))
                cond_cov = 0.5 * (cond_cov + cond_cov.T)

                # MC samples and Cholesky to turn N(0,1) to desired Gaussian distribution
                Z = rng.standard_normal((self.sample_size, len(unknown_indices)))
                samples_unknown = Z @ np.linalg.cholesky(cond_cov).T + cond_mean

                samples = np.tile(x_explain, (self.sample_size, 1))
                samples[:, unknown_indices] = samples_unknown

            samples_all_coalitions[i] = samples

        return samples_all_coalitions

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Imputes the missing values of a data point and gets predictions for all coalitions.

        Args:
            coalitions: A boolean array of shape ``(n_coalitions, n_features)`` indicating which features are present (``True``) and which are missing (``False``).

        Returns:
            The model's predictions on the imputed data points as an array of shape ``(n_coalitions,)``.

        Raises:
            RuntimeError: If no explanation point has been provided, neither in the constructor nor by calling ``fit()``.
        """
        if self.x is None:
            msg = f"Must call {self.__class__.__name__}.fit(x) first before imputing"
            raise RuntimeError(msg)

        n_coalitions = coalitions.shape[0]
        samples = self._draw_samples(self.x.flatten(), coalitions)

        predictions = np.zeros((n_coalitions, self.sample_size))
        for i in range(n_coalitions):
            predictions[i] = self.predict(samples[i])
        return cast("npt.NDArray[np.floating]", np.mean(predictions, axis=1))

    def _draw_samples(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        """Draw samples for the given coalitions to be used for computing the utility.

        This function should be overriden by a subclass, if the Monte Carlo sampling needs to be wrapped in any kind of transformation.

        Args:
            x: The explanation point as an array of shape ``(n_features,)``.
            coalitions: A set of coalitions as an array of shape ``(n_coalitions, n_features)``.

        Returns:
            Samples draw for each coalition as an array of shape ``(n_coalitions, n_samples, n_features)``.
        """
        return self._sample_monte_carlo(x, coalitions)
