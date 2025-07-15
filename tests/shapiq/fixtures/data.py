"""This fixtures module contains data fixtures for the tests.

Note to developers:
    This module should only contain fixtures containing data. If you need to creat a model fixture,
    please use the `models.py` module and import the data fixtures from this module. Further, always
    use the `copy.deepcopy` function to ensure that the fixtures are not modified during the tests.
    This is especially important for the data fixtures, as they are used in multiple tests.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import make_classification, make_regression

# normal datasets
NR_FEATURES = 7
NR_FEATURES_INFORMATIVE = 7
BUDGET_NR_FEATURES = 2**NR_FEATURES

# small datasets
NR_FEATURES_SMALL = 3
NR_FEATURES_SMALL_INFORMATIVE = 2
BUDGET_NR_FEATURES_SMALL = 2**NR_FEATURES_SMALL

DATASETS_RANDOM_STATE = 42


@pytest.fixture
def if_clf_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple dataset for the isolation forest model."""
    n_samples, n_outliers = 120, 40
    rng = np.random.RandomState(DATASETS_RANDOM_STATE)
    covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
    cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
    cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
    outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))
    X = np.concatenate([cluster_1, cluster_2, outliers])
    y = np.concatenate([np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)])
    return copy.deepcopy(X), copy.deepcopy(y)


@pytest.fixture
def background_reg_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_regression(
        n_samples=100, n_features=NR_FEATURES, random_state=DATASETS_RANDOM_STATE
    )
    return copy.deepcopy(X), copy.deepcopy(y)


@pytest.fixture
def background_reg_data(background_reg_dataset) -> np.ndarray:
    """Return a simple background dataset."""
    X, _ = background_reg_dataset
    return copy.deepcopy(X)


@pytest.fixture
def background_clf_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=NR_FEATURES,
        random_state=DATASETS_RANDOM_STATE,
        n_classes=3,
        n_informative=NR_FEATURES_INFORMATIVE,
        n_repeated=0,
        n_redundant=0,
    )
    return copy.deepcopy(X), copy.deepcopy(y)


@pytest.fixture
def background_clf_data(background_clf_dataset) -> np.ndarray:
    """Return a simple background dataset."""
    X, _ = background_clf_dataset
    return copy.deepcopy(X)


@pytest.fixture
def background_clf_dataset_binary() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=NR_FEATURES,
        random_state=DATASETS_RANDOM_STATE,
        n_classes=2,
        n_informative=NR_FEATURES_INFORMATIVE,
        n_repeated=0,
        n_redundant=0,
    )
    return copy.deepcopy(X), copy.deepcopy(y)


@pytest.fixture
def background_clf_dataset_binary_small() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_classification(
        n_samples=10,
        n_features=NR_FEATURES_SMALL,
        random_state=DATASETS_RANDOM_STATE,
        n_classes=2,
        n_informative=NR_FEATURES_SMALL_INFORMATIVE,
        n_repeated=0,
        n_redundant=NR_FEATURES_SMALL - NR_FEATURES_SMALL_INFORMATIVE,
    )
    return copy.deepcopy(X), copy.deepcopy(y)


@pytest.fixture
def background_reg_dataset_small() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_regression(
        n_samples=10, n_features=NR_FEATURES_SMALL, random_state=DATASETS_RANDOM_STATE
    )
    return copy.deepcopy(X), copy.deepcopy(y)


@pytest.fixture
def image_and_path() -> tuple[Image.Image, str]:
    """Reads and returns the test image."""
    # get path for this file's directory
    path_from_test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "test_croc.JPEG",
    )
    image = Image.open(path_from_test_root)
    return copy.deepcopy(image), copy.deepcopy(path_from_test_root)


_california_loaded_data: dict[str, np.ndarray | None] = {
    "x_train": None,
    "y_train": None,
    "x_test": None,
    "y_test": None,
    "x_explain": None,
}


def get_california_housing_train_test_explain() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load the California housing dataset and return a train and test split.

    Returns:
        tuple: A tuple containing the training data, training labels, test data, and test labels.
    """
    if all(_v is not None for _v in _california_loaded_data.values()):
        return (
            _california_loaded_data["x_train"],
            _california_loaded_data["y_train"],
            _california_loaded_data["x_test"],
            _california_loaded_data["y_test"],
            _california_loaded_data["x_explain"],
        )

    from sklearn.model_selection import train_test_split

    from shapiq.datasets import load_california_housing

    x_explain_id = 2  # index of the data point to explain (third data point of the test set)

    x_data, y_data = load_california_housing(to_numpy=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=DATASETS_RANDOM_STATE
    )
    x_explain = x_test[x_explain_id].reshape(1, -1)
    _california_loaded_data["x_train"] = copy.deepcopy(x_train)
    _california_loaded_data["y_train"] = copy.deepcopy(y_train)
    _california_loaded_data["x_test"] = copy.deepcopy(x_test)
    _california_loaded_data["y_test"] = copy.deepcopy(y_test)
    _california_loaded_data["x_explain"] = copy.deepcopy(x_explain)
    return (
        _california_loaded_data["x_train"],
        _california_loaded_data["y_train"],
        _california_loaded_data["x_test"],
        _california_loaded_data["y_test"],
        _california_loaded_data["x_explain"],
    )


@pytest.fixture
def california_housing_train_test_explain() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return the California housing dataset with a train and test split."""
    return get_california_housing_train_test_explain()
