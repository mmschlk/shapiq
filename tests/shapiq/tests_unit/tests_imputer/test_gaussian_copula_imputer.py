"""Tests for the GaussianCopulaImputer class.

This module contains unit tests for the GaussianCopulaImputer class, including tests for
categorical feature detection, transformation methods, and imputation logic.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from shapiq.imputer import GaussianCopulaImputer

##############################################
# Tests for Transformation Methods --------- #
##############################################


def test_empirical_cdf(dummy_model) -> None:
    """Tests that the empirical CDF is calculated correctly for a single column."""
    feature_values = np.array([-5, 10, 0, 3, 4])
    expected_empirical_cdf = np.array([1, 5, 2, 3, 4]) / 5

    dummy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    imputer = GaussianCopulaImputer(model=dummy_model, data=dummy_data)

    empirical_cdf = imputer._empirical_cdf(feature_values.reshape(-1, 1)).flatten()

    assert np.allclose(empirical_cdf, expected_empirical_cdf)


def test_empirical_cdf_1_dimensional(dummy_model) -> None:
    """Tests that the empirical CDF is raising error on one dimensional input."""
    feature_values = np.array([-5, 10, 0, 3, 4])

    dummy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    imputer = GaussianCopulaImputer(model=dummy_model, data=dummy_data)

    with pytest.raises(ValueError, match="Expected a 2D array"):
        imputer._empirical_cdf(feature_values)


def test_empirical_cdf_2_dimensional(dummy_model) -> None:
    """Tests that the empirical CDF is raising error on three dimensional input.."""
    feature_values = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    dummy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    imputer = GaussianCopulaImputer(model=dummy_model, data=dummy_data)

    with pytest.raises(ValueError, match="Expected a 2D array"):
        imputer._empirical_cdf(feature_values)


def test_transform_to_gaussian(dummy_model) -> None:
    """Tests transforming background data to Gaussian space."""
    data = np.array(
        [
            [1.0, 2.0, 0.0],
            [4.0, 8.0, -10.0],
            [7.0, 5.0, 200.0],
        ]
    )
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)

    expected_transformed_data = np.array(
        [
            [norm.ppf(1 / 3), norm.ppf(1 / 3), norm.ppf(2 / 3)],
            [norm.ppf(2 / 3), norm.ppf(1 - imputer.QUANTILE_CLIP_EPSILON), norm.ppf(1 / 3)],
            [
                norm.ppf(1 - imputer.QUANTILE_CLIP_EPSILON),
                norm.ppf(2 / 3),
                norm.ppf(1 - imputer.QUANTILE_CLIP_EPSILON),
            ],
        ]
    )

    transformed_data = imputer._transform_to_gaussian(data)

    assert np.allclose(transformed_data, expected_transformed_data, atol=1e-3), (
        f"Expected {expected_transformed_data}, but got {transformed_data}"
    )


def test_transform_point_to_gaussian(dummy_model) -> None:
    """Test transforming a single point to Gaussian space."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)

    x_test = np.array([8, 2, 1])
    # Features should be mapped to ranks [3, 1, 0], meaning quantiles [3/3, 1/3, 0/3]
    # The out-of-range value of 0 will be clipped according to the imputers configuration
    expected_quantiles = np.array(
        [1 - imputer.QUANTILE_CLIP_EPSILON, 1 / 3, imputer.QUANTILE_CLIP_EPSILON]
    )
    expected_x_transformed = norm.ppf(expected_quantiles)

    x_transformed = imputer._transform_point_to_gaussian(data, x_test)

    assert x_transformed.shape == x_test.shape
    assert np.allclose(x_transformed, expected_x_transformed, atol=1e-2)


def test_incorrect_transform_point_to_gaussian(dummy_model) -> None:
    """Tests Error reaction to feature number and explanation point (x) missmatch."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)

    x_test = np.array([8, 2])
    with pytest.raises(
        ValueError, match="Background data has 3 features but point to transform has 2 features"
    ):
        imputer._transform_point_to_gaussian(data, x_test)


def test_identity_transform(dummy_model) -> None:
    """Tests that transforming to Gaussian space and gives the original data."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [7.0, 2.5, 6.0],
            [2.0, 8.0, -1.0],
        ]
    )

    imputer = GaussianCopulaImputer(model=dummy_model, data=data)
    data_transformed = imputer._transform_to_gaussian(data)
    data_backtransformed = imputer._transform_from_gaussian(data_transformed)

    assert np.allclose(data, data_backtransformed)


def test_trasnform_from_gaussian(dummy_model) -> None:
    """Test transforming from Gaussian space back to original feature space."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    gaussian_samples = np.array(
        [
            [0, 0, 0],
            [1, -1, 0.5],
            [-1, 1, -0.5],
        ]
    )
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)

    original = imputer._transform_from_gaussian(gaussian_samples)

    # Check shape
    assert original.shape == gaussian_samples.shape

    # Check values are within original data range
    assert np.all(original >= np.min(data))
    assert np.all(original <= np.max(data))


##############################################
# Tests for Imputation --------------------- #
##############################################


def test_copula_imputation_single_feature_known(dummy_model) -> None:
    """Test imputation with a single feature known and two unknown."""
    # Create correlated data
    rng = np.random.default_rng(seed=42)
    data = rng.normal(size=(1000, 3))
    # Introduce correlation between features
    data[:, 1] = 0.8 * data[:, 0] + 0.2 * data[:, 1]
    data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 2]

    x_explain = np.array([1.0, np.nan, np.nan])
    coalitions = np.array([[True, False, False]])

    imputer = GaussianCopulaImputer(
        model=dummy_model, data=data, x=x_explain, sample_size=1000, random_state=42
    )

    samples = imputer._draw_samples(x_explain, coalitions)

    assert samples.shape == (coalitions.shape[0], imputer.sample_size, x_explain.shape[0])
    samples_avg = np.mean(samples, axis=1)

    # We can't predict exact values, but they should be within a reasonable range on average
    lower_bound = -2
    upper_bound = 4
    assert np.all(samples_avg[0, 1:] >= lower_bound)
    assert np.all(samples_avg[0, 1:] <= upper_bound)


def test_copula_imputer_value_function(dummy_model) -> None:
    """Test the value function of the copula imputer."""
    # Create correlated data
    rng = np.random.default_rng(seed=42)
    data = rng.normal(size=(1000, 3))
    # Introduce correlation between features
    data[:, 1] = 0.8 * data[:, 0] + 0.2 * data[:, 1]
    data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 2]

    x_explain = np.array([1.0, np.nan, np.nan])
    coalitions = np.array([[True, False, False]])

    imputer = GaussianCopulaImputer(
        model=dummy_model, data=data, x=x_explain, sample_size=1000, random_state=42
    )
    y_predicted = imputer.value_function(coalitions)

    assert y_predicted.shape == (coalitions.shape[0],)
    # The expected value should be sum of known feature (1.0) plus the conditional means
    # We can't predict exact values, but they should be within a reasonable range
    imputed_lower = -2
    imputed_upper = 4
    assert np.all(y_predicted[0] >= imputed_lower)
    assert np.all(y_predicted[0] <= imputed_upper)
