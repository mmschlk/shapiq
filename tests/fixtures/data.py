"""This fixtures module contains data fixtures for the tests."""

import os

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import make_classification, make_regression

NR_FEATURES = 7  # Number of features for the tabular models


@pytest.fixture
def if_clf_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple dataset for the isolation forest model."""
    n_samples, n_outliers = 120, 40
    rng = np.random.RandomState(0)
    covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
    cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
    cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
    outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))
    X = np.concatenate([cluster_1, cluster_2, outliers])
    y = np.concatenate([np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)])
    return X, y


@pytest.fixture
def background_reg_data() -> np.ndarray:
    """Return a simple background dataset."""
    X, _ = make_regression(n_samples=100, n_features=7, random_state=42)
    return X


@pytest.fixture
def background_clf_data() -> np.ndarray:
    """Return a simple background dataset."""
    X, _ = make_classification(
        n_samples=100,
        n_features=NR_FEATURES,
        random_state=42,
        n_classes=3,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
    )
    return X


@pytest.fixture
def background_reg_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_regression(n_samples=100, n_features=NR_FEATURES, random_state=42)
    return X, y


@pytest.fixture
def background_clf_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=NR_FEATURES,
        random_state=42,
        n_classes=3,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
    )
    return X, y


@pytest.fixture
def background_clf_dataset_binary() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=NR_FEATURES,
        random_state=42,
        n_classes=2,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
    )
    return X, y


@pytest.fixture
def image_and_path() -> tuple[Image.Image, str]:
    """Reads and returns the test image."""
    # get path for this file's directory
    path_from_test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "test_croc.JPEG"
    )
    image = Image.open(path_from_test_root)
    return image, path_from_test_root
