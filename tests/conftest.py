"""This test module contains all fixtures for all tests of shapiq.
If it becomes too large, it can be split into multiple files like here:
https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
"""
import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


@pytest.fixture
def dt_reg_model() -> DecisionTreeRegressor:
    """Return a simple decision tree model."""
    X, y = make_regression(n_samples=100, n_features=7, random_state=42)
    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def dt_clf_model() -> DecisionTreeClassifier:
    """Return a simple decision tree model."""
    X, y = make_classification(
        n_samples=100,
        n_features=7,
        random_state=42,
        n_classes=3,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
    )
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def rf_reg_model() -> RandomForestRegressor:
    """Return a simple random forest model."""
    X, y = make_regression(n_samples=100, n_features=7, random_state=42)
    model = RandomForestRegressor(random_state=42, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def rf_clf_model() -> RandomForestClassifier:
    """Return a simple random forest model."""
    X, y = make_classification(
        n_samples=100,
        n_features=7,
        random_state=42,
        n_classes=3,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
    )
    model = RandomForestClassifier(random_state=42, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def background_reg_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_regression(n_samples=100, n_features=7, random_state=42)
    return X


@pytest.fixture
def background_clf_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=7,
        random_state=42,
        n_classes=3,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
    )
    return X
