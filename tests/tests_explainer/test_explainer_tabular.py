"""This test module contains all tests regarding the interaciton explainer for the shapiq package."""

import pytest
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from shapiq.approximator import RegressionFSII
from shapiq.explainer import TabularExplainer

import numpy as np

@pytest.fixture
def dt_model():
    """Return a simple decision tree model."""
    X, y = make_regression(n_samples=100, n_features=7, random_state=42)
    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def data():
    """Return data to use as background data."""
    X, y = make_regression(n_samples=100, n_features=7, random_state=42)
    return X


INDICES = ["SII", "k-SII", "STII", "FSII"]
MAX_ORDERS = [2, 3]
IMPUTER = ["marginal", "conditional"]


@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("max_order", MAX_ORDERS)
@pytest.mark.parametrize("imputer", IMPUTER)
def test_init_params(dt_model, data, index, max_order, imputer):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        random_state=42,
        index=index,
        max_order=max_order,
        approximator="auto",
        imputer=imputer,
    )
    assert explainer.index == index
    assert explainer._approximator.index == index
    assert explainer._max_order == max_order
    assert explainer._random_state == 42
    # test defaults
    if index == "FSII":
        assert explainer._approximator.__class__.__name__ == "RegressionFSII"
    elif index == "SII" or index == "k-SII":
        assert explainer._approximator.__class__.__name__ == "KernelSHAPIQ"
    else:
        assert explainer._approximator.__class__.__name__ == "SHAPIQ"


def test_auto_params(dt_model, data):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    explainer = TabularExplainer(
        model=model_function,
        data=data,
    )
    assert explainer.index == "k-SII"
    assert explainer._approximator.index == "k-SII"
    assert explainer._max_order == 2
    assert explainer._random_state is None
    assert explainer._approximator.__class__.__name__ == "KernelSHAPIQ"


def test_init_params_error(dt_model, data):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    with pytest.raises(ValueError):
        TabularExplainer(
            model=model_function,
            data=data,
            index="invalid",
        )
    with pytest.raises(ValueError):
        TabularExplainer(
            model=model_function,
            data=data,
            max_order=0,
        )


def test_init_params_approx(dt_model, data):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    with pytest.raises(ValueError):
        TabularExplainer(
            model=model_function,
            data=data,
            approximator="invalid",
        )
    explainer = TabularExplainer(
        approximator="Regression",
        index="FSII",
        model=model_function,
        data=data,
    )
    assert explainer._approximator.__class__.__name__ == "RegressionFSII"

    # init explainer with manual approximator
    approximator = RegressionFSII(n=9, max_order=2)
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        approximator=approximator,
    )
    assert explainer._approximator.__class__.__name__ == "RegressionFSII"
    assert explainer._approximator == approximator


BUDGETS = [2**5, 2**8, None]


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("max_order", MAX_ORDERS)
@pytest.mark.parametrize("imputer", IMPUTER)
def test_explain(dt_model, data, index, budget, max_order, imputer):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        random_state=42,
        index=index,
        max_order=max_order,
        approximator="auto",
        imputer=imputer,
    )
    x = data[0].reshape(1, -1)
    interaction_values = explainer.explain(x, budget=budget)
    assert interaction_values.index == index
    assert interaction_values.max_order == max_order
    if budget is None:
        budget = 100_000_000_000
    assert interaction_values.estimation_budget <= budget + 2
    interaction_values0 = explainer.explain(x, budget=budget, random_state=0)
    interaction_values2 = explainer.explain(x, budget=budget, random_state=0)
    assert np.allclose(
        interaction_values0.get_n_order_values(1),
        interaction_values2.get_n_order_values(1)
    )
    assert np.allclose(
        interaction_values0.get_n_order_values(2),
        interaction_values2.get_n_order_values(2)
    )
