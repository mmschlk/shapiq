"""This fixtures module contains model fixtures for the tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.explainer.tree import TreeModel
from shapiq.utils import Model

NR_FEATURES = 7  # Number of features for the tabular models

TABULAR_MODEL_FIXTURES = [
    ("custom_model", "custom_model"),
    ("lr_reg_model", "sklearn.linear_model.LinearRegression"),
    ("lr_clf_model", "sklearn.linear_model.LogisticRegression"),
]

TABULAR_TENSORFLOW_MODEL_FIXTURES = [
    ("sequential_model_1_class", "tensorflow.python.keras.engine.sequential.Sequential"),
    ("sequential_model_2_classes", "keras.src.models.sequential.Sequential"),
    ("sequential_model_3_classes", "keras.engine.sequential.Sequential"),
]

TABULAR_TORCH_MODEL_FIXTURES = [
    ("torch_clf_model", "torch.nn.modules.container.Sequential"),
    ("torch_reg_model", "torch.nn.modules.container.Sequential"),
]

TREE_MODEL_FIXTURES = [
    ("xgb_reg_model", "xgboost.sklearn.XGBRegressor"),
    ("xgb_clf_model", "xgboost.sklearn.XGBClassifier"),
    ("lightgbm_reg_model", "lightgbm.sklearn.LGBMRegressor"),
    ("lightgbm_clf_model", "lightgbm.sklearn.LGBMClassifier"),
    ("lightgbm_basic", "lightgbm.basic.Booster"),
    ("rf_reg_model", "sklearn.ensemble.RandomForestRegressor"),
    ("rf_clf_model", "sklearn.ensemble.RandomForestClassifier"),
    ("dt_clf_model", "sklearn.tree.DecisionTreeClassifier"),
    ("dt_reg_model", "sklearn.tree.DecisionTreeRegressor"),
]


class CustomModel:
    def __init__(self, data: tuple[np.ndarray, np.ndarray]):
        self.data = data

    def __call__(self, *args, **kwargs):
        return self.data[1]


@pytest.fixture
def custom_model(background_reg_dataset) -> CustomModel:
    """Return a callable mock custom model"""
    return CustomModel(background_reg_dataset)


@pytest.fixture
def lightgbm_basic(background_reg_dataset) -> Model:
    """Return a lgm basic booster"""
    lightgbm = pytest.importorskip("lightgbm")

    X, y = background_reg_dataset
    train_data = lightgbm.Dataset(X, label=y)
    model = lightgbm.train(params={}, train_set=train_data, num_boost_round=1)
    return model


@pytest.fixture
def sequential_model_1_class() -> Model:
    """Return a keras nn with output dimension 1"""
    return _sequential_model(1)


@pytest.fixture
def sequential_model_2_classes() -> Model:
    """Return a keras nn with output dimension 2"""
    return _sequential_model(2)


@pytest.fixture
def sequential_model_3_classes() -> Model:
    """Return a keras nn with output dimension 3"""
    return _sequential_model(3)


def _sequential_model(output_shape_nr, background_reg_dataset) -> Model:
    """Return a keras nn with specified output dimension"""
    keras = pytest.importorskip("keras")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(NR_FEATURES,)),
            keras.layers.Dense(2, activation="relu", name="layer1"),
            keras.layers.Dense(output_shape_nr, name="layer2"),
        ],
    )
    model.compile(optimizer="adam", loss="mse")
    X, y = background_reg_dataset
    model.fit(X, y, epochs=0, batch_size=32)
    return model


@pytest.fixture
def xgb_reg_model(background_reg_dataset) -> Model:
    """Return a simple xgboost regression model."""
    xgboost = pytest.importorskip("xgboost")

    X, y = background_reg_dataset
    model = xgboost.XGBRegressor(random_state=42, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def rf_clf_binary_model(background_clf_dataset_binary) -> RandomForestClassifier:
    """Return a simple random forest model."""
    X, y = background_clf_dataset_binary
    model = RandomForestClassifier(random_state=42, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def xgb_clf_model(background_clf_dataset) -> Model:
    """Return a simple xgboost classification model."""
    xgboost = pytest.importorskip("xgboost")

    X, y = background_clf_dataset
    model = xgboost.XGBClassifier(random_state=42, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def torch_clf_model() -> Model:
    """Return a simple torch model."""
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(
        torch.nn.Linear(7, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 3),
    )
    model.eval()
    return model


@pytest.fixture
def torch_reg_model() -> Model:
    """Return a simple torch model."""
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(
        torch.nn.Linear(7, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    model.eval()
    return model


@pytest.fixture
def tabpfn_classification_problem(
    background_clf_dataset_binary_small,
) -> tuple[Model, np.ndarray, np.ndarray, np.ndarray]:
    """Returns a very simple tabpfn classifier and dataset."""
    tabpfn = pytest.importorskip("tabpfn")

    data, labels = background_clf_dataset_binary_small
    data, x_test, labels, _ = train_test_split(data, labels, random_state=42, train_size=8)
    model = tabpfn.TabPFNClassifier()
    model.fit(data, labels)
    return model, data, labels, x_test


@pytest.fixture
def tabpfn_regression_problem(
    background_reg_dataset_small,
) -> tuple[Model, np.ndarray, np.ndarray, np.ndarray]:
    """Returns a very simple tabpfn regressor and dataset."""
    tabpfn = pytest.importorskip("tabpfn")

    data, labels = background_reg_dataset_small
    data, x_test, labels, _ = train_test_split(data, labels, random_state=42, train_size=8)
    model = tabpfn.TabPFNRegressor()
    model.fit(data, labels)
    return model, data, labels, x_test


@pytest.fixture
def dt_reg_model(background_reg_dataset) -> DecisionTreeRegressor:
    """Return a simple decision tree model."""
    X, y = background_reg_dataset
    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def dt_clf_model(background_clf_dataset) -> DecisionTreeClassifier:
    """Return a simple decision tree model."""
    X, y = background_clf_dataset
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def lr_clf_model(background_clf_dataset) -> LogisticRegression:
    """Return a simple logistic regression model."""
    X, y = background_clf_dataset
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model


@pytest.fixture
def lr_reg_model(background_reg_dataset) -> LinearRegression:
    """Return a simple linear regression model."""
    X, y = background_reg_dataset
    model = LinearRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def lightgbm_reg_model(background_reg_dataset) -> Model:
    """Return a simple lightgbm regression model."""
    lightgbm = pytest.importorskip("lightgbm")

    X, y = background_reg_dataset
    model = lightgbm.LGBMRegressor(random_state=42, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def lightgbm_clf_model(background_clf_dataset) -> Model:
    """Return a simple lightgbm classification model."""
    lightgbm = pytest.importorskip("lightgbm")

    X, y = background_clf_dataset
    model = lightgbm.LGBMClassifier(random_state=42, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def dt_clf_model_tree_model(background_clf_dataset) -> TreeModel:
    """Return a simple decision tree as a TreeModel."""
    from shapiq.explainer.tree.validation import validate_tree_model

    X, y = background_clf_dataset
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X, y)
    tree_model = validate_tree_model(model)
    return tree_model


@pytest.fixture
def rf_reg_model(background_reg_dataset) -> RandomForestRegressor:
    """Return a simple random forest model."""
    X, y = background_reg_dataset
    model = RandomForestRegressor(random_state=42, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def rf_clf_model(background_clf_dataset) -> RandomForestClassifier:
    """Return a simple (classification) random forest model."""
    X, y = background_clf_dataset
    model = RandomForestClassifier(random_state=42, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


# Isolationforest model
@pytest.fixture
def if_clf_model(if_clf_dataset) -> IsolationForest:
    """Return a simple isolation forest model."""
    X, y = if_clf_dataset
    model = IsolationForest(random_state=42, n_estimators=3)
    model.fit(X, y)
    return model


# Extra trees model
@pytest.fixture
def et_clf_model(background_clf_dataset) -> Model:
    """Return a simple (classification) extra trees model."""

    X, y = background_clf_dataset
    model = ExtraTreesClassifier(random_state=42, max_depth=3, n_estimators=3, verbose=False)
    model.fit(X, y)
    return model


@pytest.fixture
def et_reg_model(background_reg_dataset) -> Model:
    """Return a simple (regression) extra trees model."""

    X, y = background_reg_dataset
    model = ExtraTreesRegressor(random_state=42, max_depth=3, n_estimators=3, verbose=False)
    model.fit(X, y)
    return model


# CatBoost model
@pytest.fixture
def cb_clf_model(background_clf_dataset) -> Model:
    """Return a simple CatBoost classification model."""
    catboost = pytest.importorskip("catboost")

    X, y = background_clf_dataset
    model = catboost.CatBoostClassifier(depth=3, random_state=42, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def cb_reg_model(background_reg_dataset) -> Model:
    """Return a simple CatBoost regression model."""
    catboost = pytest.importorskip("catboost")

    X, y = background_reg_dataset
    model = catboost.CatBoostRegressor(depth=3, random_state=42, n_estimators=3)
    model.fit(X, y)
    return model
