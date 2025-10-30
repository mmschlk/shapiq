"""Implements the Gaussian copula-based approach for imputation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.stats import norm, rankdata

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from shapiq.game import Game

from .gaussian_imputer import GaussianImputer


class GaussianCopulaImputer(GaussianImputer):
    r"""Implements the Gaussian copula-based approach for imputation according to [Aas21]_.

    This method models feature dependence using a Gaussian copula, separating the modeling of marginal distributions
    from their joint dependence structure. Each feature is first transformed to follow a standard normal distribution by
    applying the empirical cumulative distribution function (CDF), followed by the standard normal quantile function.

    Monte Carlo samples are then drawn from the conditional multivariate normal distribution in this transformed (Gaussian)
    space, given the observed values in a coalition. These samples are finally transformed back to the original feature space
    using the inverse of the empirical CDFs.
    """

    QUANTILE_CLIP_EPSILON = 1e-10
    """Used for clipping values to the 'exclusive range' ``(0, 1)`` when evaluating a quantile function.

    More specifically, values will be clipped to the range ``[epsilon, 1 - epsilon]``."""

    def __init__(
        self,
        model: (object | Game | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]),
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        sample_size: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the GaussianCopulaImputer.

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
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            random_state=random_state,
            verbose=verbose,
        )

        self._data_transformed = self._transform_to_gaussian(self.data)
        self._mean_per_feature = np.mean(self._data_transformed, axis=0)
        self._cov_mat = self._ensure_positive_definite(np.cov(self._data_transformed.T))
        # Sorted data is required for the transformation back from Gaussian space to the original feature space
        self._data_sorted = np.sort(self.data, axis=0)

    def _transform_to_gaussian(
        self, background_data: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform each feature to a standard normal distribution using empirical CDF (rank-Gaussian).

        For each feature, this method applies a transformation such that the values follow a standard normal
        distribution, while preserving the rank order of the original data. This is also known as a
        rank-Gaussian or empirical CDF transformation.

        Args:
            background_data: Input data to transform as an array of shape ``(n_samples, n_features)``.

        Returns:
            Transformed data in Gaussian space as an array of shape ``(n_samples, n_features)``.
        """
        empirical_cdf = self._empirical_cdf(background_data)
        empirical_cdf = np.clip(
            empirical_cdf, self.QUANTILE_CLIP_EPSILON, 1 - self.QUANTILE_CLIP_EPSILON
        )
        return cast("npt.NDArray[np.floating]", norm.ppf(empirical_cdf))

    def _empirical_cdf(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Computes the empirical cumulative distribution function for each feature of the given training data matrix.

        Each column of the input is treated as samples drawn from a separate random variable and transformed to its empirical CDF.

        Args:
            data: An array of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples, n_features)` containing the CDF for each column.
        """
        expected_dimensionality = 2
        if data.ndim == 1:
            msg = "Expected a 2D array but got a 1D array. To compute the empirical CDF for a single feature, use array.reshape(-1, 1)"
            raise ValueError(msg)
        if data.ndim > expected_dimensionality:
            msg = f"Expected a 2D array but got a {data.ndim}-dimensional array instead."
            raise ValueError(msg)

        ranks = rankdata(data, axis=0, method="average")
        return ranks / data.shape[0]

    def _transform_point_to_gaussian(
        self,
        background_data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Transforms a single data point to Gaussian space using the training data's empirical CDF.

        Args:
            background_data: Data used to compute the empirical CDF, with shape ``(n_samples, n_features)``.
            x: Data point to transform, with shape ``(n_features,)``.

        Returns:
            Transformed point in Gaussian space, with shape ``(n_features,)``.
        """
        n_features = background_data.shape[1]
        if x.shape[0] != n_features:
            msg = f"Background data has {n_features} features but point to transform has {x.shape[0]} features"
            raise ValueError(msg)

        x_empirical_cdf = self._empirical_cdf_point(background_data, x)
        x_empirical_cdf = np.clip(
            x_empirical_cdf, self.QUANTILE_CLIP_EPSILON, 1 - self.QUANTILE_CLIP_EPSILON
        )
        return cast("npt.NDArray[np.floating]", norm.ppf(x_empirical_cdf))

    def _empirical_cdf_point(
        self, data: npt.NDArray[np.floating], x: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Evaluates an empirical cumulative distribution function for each feature of a single data point.

        Args:
            data: The background data as an array of shape `(n_samples, n_features)` which defines an empirical CDF for each feature.
            x: A data point for which to evaluate CDFs for.

        Returns:
            An array of shape `(n_features)` containing the CDF of each feature evaluated for the value of ``x`` in that column.
        """
        n_samples = data.shape[0]
        ranks = cast("npt.NDArray[np.integer]", np.sum(data <= x, axis=0))
        return ranks / n_samples

    def _transform_from_gaussian(
        self, data_gaussian: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform Gaussian samples back to original feature space.

        Args:
            data_gaussian: Transformed samples in Gaussian space as an array of ``(n_coalitions, n_samples, n_features)`` or ``(n_samples, n_features)``.
            If a 2D array is provided, it is assumed that it corresponds to one coalition.

        Returns:
            Samples in original feature spaces as an array of ``(n_coalitions, n_samples, n_features)`` or ``(n_samples, n_features)``.
            The shape corresponds to the input shape.
        """
        input_2d = False
        if data_gaussian.ndim != 3:
            # We assume that the input corresponds to one coalition
            data_gaussian = data_gaussian[np.newaxis, :, :]
            input_2d = True
        n_features = data_gaussian.shape[2]
        n_samples = self.data.shape[0]

        quantiles = norm.cdf(data_gaussian)  # shape (n_coalitions, n_samples, n_features)
        ranks = quantiles * n_samples

        x_original = np.zeros_like(data_gaussian)
        rank_indices = np.arange(1, n_samples + 1)

        for col in range(n_features):
            # The back-transformed ranks are not necessarily integers, so we interpolate linearly between the closest original datapoints
            _data_sorted = self._data_sorted[:, col]  # Better to access it once outside the loop
            x_original[:, :, col] = np.apply_along_axis(
                lambda r, _data_sorted=_data_sorted: np.interp(
                    r,
                    rank_indices,
                    _data_sorted,
                ),
                0,
                ranks[:, :, col],
            )

        if input_2d:
            # Return to original shape
            return x_original[0]
        return x_original

    def _draw_samples(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        """Draw samples for the given coalitions to be used for computing the utility.

        The explanation point ``x`` is first transformed to Gaussian space. Then, samples are drawn,
        and finally the samples are transformed back to the original feature space.

        Args:
            x: The explanation point as an array of shape ``(n_features,)``.
            coalitions: A set of coalitions as an array of shape ``(n_coalitions, n_features)``.

        Returns:
            Samples draw for each coalition as an array of shape ``(n_coalitions, n_samples, n_features)``.
        """
        x_transformed = self._transform_point_to_gaussian(self.data, x)

        samples_gaussian = self._sample_monte_carlo(x_transformed, coalitions)

        return self._transform_from_gaussian(samples_gaussian)
