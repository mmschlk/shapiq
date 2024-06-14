"""This test module contains all fixtures for all tests of shapiq.
If it becomes too large, it can be split into multiple files like here:
https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
"""

import os

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.explainer.tree import TreeModel
from shapiq.interaction_values import InteractionValues


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
def dt_clf_model_tree_model() -> TreeModel:
    """Return a simple decision tree as a TreeModel."""
    from shapiq.explainer.tree.validation import validate_tree_model

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
    tree_model = validate_tree_model(model)
    return tree_model


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
def rf_clf_binary_model() -> RandomForestClassifier:
    """Return a simple random forest model."""
    X, y = make_classification(
        n_samples=100,
        n_features=7,
        random_state=42,
        n_classes=2,
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


@pytest.fixture
def background_reg_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_regression(n_samples=100, n_features=7, random_state=42)
    return X, y


@pytest.fixture
def background_clf_dataset() -> tuple[np.ndarray, np.ndarray]:
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
    return X, y


@pytest.fixture
def background_clf_dataset_binary() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple background dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=7,
        random_state=42,
        n_classes=2,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
    )
    return X, y


@pytest.fixture
def test_image_and_path() -> tuple[Image.Image, str]:
    """Reads and returns the test image."""
    # get path for this file's directory
    path_from_test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "test_croc.JPEG"
    )
    image = Image.open(path_from_test_root)
    return image, path_from_test_root


@pytest.fixture
def mae_loss():
    """Returns the mean absolute error loss function."""
    from sklearn.metrics import mean_absolute_error

    return mean_absolute_error


@pytest.fixture
def interaction_values_list() -> list[InteractionValues]:
    """Returns a list of three InteractionValues objects."""
    from shapiq.utils import powerset

    n_objects = 3
    n_players = 5
    min_order = 0
    max_order = n_players
    iv_list = []
    for i in range(n_objects):
        values = []
        interaction_lookup = {}
        for i, interaction in enumerate(
            powerset(range(n_players), min_size=min_order, max_size=max_order)
        ):
            interaction_lookup[interaction] = i
            values.append(np.random.rand())
        values = np.array(values)
        iv = InteractionValues(
            n_players=n_players,
            values=values,
            baseline_value=float(values[interaction_lookup[tuple()]]),
            index="Moebius",
            interaction_lookup=interaction_lookup,
            max_order=max_order,
            min_order=min_order,
        )
        iv_list.append(iv)
    return iv_list
