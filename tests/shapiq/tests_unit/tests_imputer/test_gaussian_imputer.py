"""Tests for the GaussianImputer class."""

from __future__ import annotations

from typing import cast

import numpy as np

from shapiq.imputer import GaussianImputer

##############################################
# Tests for Cov Mat and Mean Calculation --- #
##############################################


def test_calculate_mean_per_feature(dummy_model) -> None:
    """Test that the mean per feature is calculated correctly."""
    rng = np.random.default_rng(seed=456)
    n_samples = 200
    n_features = 10
    data = rng.normal(size=(n_samples, n_features))
    x = np.zeros(n_features)

    imputer = GaussianImputer(model=dummy_model, data=data, x=x)
    expected_mean = np.mean(data, axis=0)

    np.testing.assert_allclose(imputer.mean_per_feature, expected_mean)


def test_calculate_covariance_matrix(dummy_model) -> None:
    """Test that the covariance matrix is calculated correctly."""
    rng = np.random.default_rng(seed=456)
    n_samples = 200
    n_features = 10
    data = rng.normal(size=(n_samples, n_features))
    x = np.array([2.0, 3.0, 4.0])

    imputer = GaussianImputer(model=dummy_model, data=data, x=x)
    expected_cov = np.cov(data.T)
    np.testing.assert_allclose(imputer.cov_mat, expected_cov)


##############################################
# Tests for Imputation ---                    #
##############################################


def test_gaussian_imputation_single_feature_known(dummy_model) -> None:
    """Test the imputation for a coalition with one known and two unknown features."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array(
        [
            [1, 0.8, 0.5],
            [0.8, 1, 0.3],
            [0.5, 0.3, 1],
        ]
    )

    rng = np.random.default_rng(seed=42)
    x_train = rng.multivariate_normal(mean, cov, size=10000)
    x_explain = np.array([1.0, np.nan, np.nan])
    coalitions = np.array([[True, False, False]])

    imputer = GaussianImputer(
        model=dummy_model,
        data=x_train,
        x=x_explain,
        sample_size=1000,
    )
    samples = imputer._draw_samples(x_explain, coalitions)
    assert samples.shape == (coalitions.shape[0], imputer.sample_size, mean.shape[0])

    samples_avg = np.mean(samples[0, :, ~coalitions[0]], axis=1)

    np.testing.assert_allclose(samples_avg, [0.8, 0.5], atol=0.1)


def test_gaussian_imputer_value_function(dummy_model):
    """Tests that the value function of the Gaussian imputer gives the expected result using a dummy model."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array(
        [
            [1, 0.8, 0.5],
            [0.8, 1, 0.3],
            [0.5, 0.3, 1],
        ]
    )
    x_explain = np.array([[1.0, np.nan, np.nan]])
    coalition = np.array([True, False, False])

    expected_imputed = np.array([1.0, 0.8, 0.5])
    y_expected = cast("np.floating", dummy_model(expected_imputed))

    rng = np.random.default_rng(seed=42)
    x_train = rng.multivariate_normal(mean, cov, size=10000)

    imputer = GaussianImputer(data=x_train, x=x_explain[0], model=dummy_model, sample_size=1000)
    y_predicted = imputer.value_function(np.atleast_2d(coalition))

    np.testing.assert_allclose(y_predicted, y_expected, atol=0.1)
