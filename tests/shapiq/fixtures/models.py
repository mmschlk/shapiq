"""This fixtures module contains model fixtures for the tests."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

if TYPE_CHECKING:
    import numpy as np

    from shapiq.explainer.tree import TreeModel
    from shapiq.utils import Model

RANDOM_SEED_MODELS = 42

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

PRODUCT_KERNEL_MODEL_FIXTURES = [
    ("bin_svc_model", "sklearn.svm.SVC"),
    ("svr_model", "sklearn.svm.SVR"),
    ("gp_reg_model", "sklearn.gaussian_process.GaussianProcessRegressor"),
]


class CustomModel:
    """A mock custom model that returns the second element of the dataset when called."""

    def __init__(self, data: tuple[np.ndarray, np.ndarray]):
        """Initialize the custom model with the dataset."""
        self.data = data

    def __call__(self, *args, **kwargs):
        """Call the model to return the second element of the dataset."""
        return self.data[1]


@pytest.fixture
def custom_model(background_reg_dataset) -> CustomModel:
    """Return a callable mock custom model."""
    return CustomModel(background_reg_dataset)


@pytest.fixture
def lightgbm_basic(background_reg_dataset) -> Model:
    """Return a lgm basic booster."""
    lightgbm = pytest.importorskip("lightgbm")

    X, y = background_reg_dataset
    train_data = lightgbm.Dataset(X, label=y)
    return lightgbm.train(params={}, train_set=train_data, num_boost_round=1)


@pytest.fixture
def sequential_model_1_class() -> Model:
    """Return a keras nn with output dimension 1."""
    return _sequential_model(1)


@pytest.fixture
def sequential_model_2_classes() -> Model:
    """Return a keras nn with output dimension 2."""
    return _sequential_model(2)


@pytest.fixture
def sequential_model_3_classes() -> Model:
    """Return a keras nn with output dimension 3."""
    return _sequential_model(3)


def _sequential_model(output_shape_nr, background_reg_dataset) -> Model:
    """Return a keras nn with specified output dimension."""
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
    model = xgboost.XGBRegressor(random_state=RANDOM_SEED_MODELS, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def rf_clf_binary_model(background_clf_dataset_binary) -> RandomForestClassifier:
    """Return a simple random forest model."""
    X, y = background_clf_dataset_binary
    model = RandomForestClassifier(random_state=RANDOM_SEED_MODELS, max_depth=3, n_estimators=3)
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
    model = tabpfn.TabPFNClassifier(n_estimators=1, fit_mode="low_memory")
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
    model = tabpfn.TabPFNRegressor(n_estimators=1, fit_mode="low_memory")
    model.fit(data, labels)
    return model, data, labels, x_test


@pytest.fixture
def dt_reg_model(background_reg_dataset) -> DecisionTreeRegressor:
    """Return a simple decision tree model."""
    X, y = background_reg_dataset
    model = DecisionTreeRegressor(random_state=RANDOM_SEED_MODELS, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def dt_clf_model(background_clf_dataset) -> DecisionTreeClassifier:
    """Return a simple decision tree model."""
    X, y = background_clf_dataset
    model = DecisionTreeClassifier(random_state=RANDOM_SEED_MODELS, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def lr_clf_model(background_clf_dataset) -> LogisticRegression:
    """Return a simple logistic regression model."""
    X, y = background_clf_dataset
    model = LogisticRegression(random_state=RANDOM_SEED_MODELS, max_iter=200)
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
    model = lightgbm.LGBMRegressor(random_state=RANDOM_SEED_MODELS, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def lightgbm_clf_model(background_clf_dataset) -> Model:
    """Return a simple lightgbm classification model."""
    lightgbm = pytest.importorskip("lightgbm")

    X, y = background_clf_dataset
    model = lightgbm.LGBMClassifier(random_state=RANDOM_SEED_MODELS, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def dt_clf_model_tree_model(background_clf_dataset) -> TreeModel:
    """Return a simple decision tree as a TreeModel."""
    from shapiq.explainer.tree.validation import validate_tree_model

    X, y = background_clf_dataset
    model = DecisionTreeClassifier(random_state=RANDOM_SEED_MODELS, max_depth=3)
    model.fit(X, y)
    return validate_tree_model(model)


@pytest.fixture
def rf_reg_model(background_reg_dataset) -> RandomForestRegressor:
    """Return a simple random forest model."""
    X, y = background_reg_dataset
    model = RandomForestRegressor(random_state=RANDOM_SEED_MODELS, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def rf_clf_model(background_clf_dataset) -> RandomForestClassifier:
    """Return a simple (classification) random forest model."""
    X, y = background_clf_dataset
    model = RandomForestClassifier(random_state=RANDOM_SEED_MODELS, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


# Isolationforest model
@pytest.fixture
def if_clf_model(if_clf_dataset) -> IsolationForest:
    """Return a simple isolation forest model."""
    X, y = if_clf_dataset
    model = IsolationForest(random_state=RANDOM_SEED_MODELS, n_estimators=3)
    model.fit(X, y)
    return model


# Extra trees model
@pytest.fixture
def et_clf_model(background_clf_dataset) -> Model:
    """Return a simple (classification) extra trees model."""
    X, y = background_clf_dataset
    model = ExtraTreesClassifier(random_state=RANDOM_SEED_MODELS, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def et_reg_model(background_reg_dataset) -> Model:
    """Return a simple (regression) extra trees model."""
    X, y = background_reg_dataset
    model = ExtraTreesRegressor(random_state=RANDOM_SEED_MODELS, max_depth=3, n_estimators=3)
    model.fit(X, y)
    return model


@pytest.fixture
def bin_svc_model(background_clf_dataset_binary) -> Model:
    """Return a simple binary SVC model."""
    X, y = background_clf_dataset_binary
    model = SVC(kernel="rbf", random_state=RANDOM_SEED_MODELS, probability=False)
    model.fit(X, y)
    return model


@pytest.fixture
def svr_model(background_reg_dataset) -> Model:
    """Return a simple SVR model."""
    X, y = background_reg_dataset
    model = SVR(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X, y)
    return model


@pytest.fixture
def gp_reg_model(background_reg_dataset) -> GaussianProcessRegressor:
    """Return a simple Gaussian Process Regressor model."""
    X, y = background_reg_dataset
    kernel = RBF(length_scale=1.0)
    model = GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_SEED_MODELS)
    model.fit(X, y)
    return model


_CALIFORNIA_HOUSING_MODEL: dict[Literal["model"], RandomForestRegressor | None] = {"model": None}


def get_california_housing_random_forest() -> RandomForestRegressor:
    """Return a random forest model trained on the California housing dataset."""
    # check if the model is already cached
    model = _CALIFORNIA_HOUSING_MODEL["model"]
    if isinstance(model, RandomForestRegressor):
        return copy.deepcopy(model)
    # fit and cache the model
    from .data import get_california_housing_train_test_explain

    x_train, y_train, _, _, _ = get_california_housing_train_test_explain()
    model = RandomForestRegressor(random_state=RANDOM_SEED_MODELS, n_estimators=10, max_depth=10)
    model.fit(x_train, y_train)
    _CALIFORNIA_HOUSING_MODEL["model"] = copy.deepcopy(model)  # cache model for quick access later
    return model


@pytest.fixture
def california_housing_rf_model() -> RandomForestRegressor:
    """Return a random forest model trained on the California housing dataset."""
    return get_california_housing_random_forest()
