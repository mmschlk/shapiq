"""This test module contains all tests regarding the interaciton explainer for the shapiq package."""

import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from shapiq.approximator import RegressionFSI
from shapiq.explainer import TabularExplainer


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


INDICES = ["SII", "k-SII", "STI", "FSI"]
MAX_ORDERS = [2, 3]


@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("max_order", MAX_ORDERS)
def test_init_params(dt_model, data, index, max_order):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        random_state=42,
        index=index,
        max_order=max_order,
        approximator="auto",
    )
    assert explainer.index == index
    assert explainer._approximator.index == index
    assert explainer._max_order == max_order
    assert explainer._random_state == 42
    # test defaults
    if index == "FSI":
        assert explainer._approximator.__class__.__name__ == "RegressionFSI"
    else:
        assert explainer._approximator.__class__.__name__ == "ShapIQ"


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
    assert explainer._approximator.__class__.__name__ == "ShapIQ"


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
        index="FSI",
        model=model_function,
        data=data,
    )
    assert explainer._approximator.__class__.__name__ == "RegressionFSI"

    # init explainer with manual approximator
    approximator = RegressionFSI(n=9, max_order=2)
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        approximator=approximator,
    )
    assert explainer._approximator.__class__.__name__ == "RegressionFSI"
    assert explainer._approximator == approximator


BUDGETS = [2**5, 2**8, None]


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("max_order", MAX_ORDERS)
def test_explain(dt_model, data, index, budget, max_order):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        random_state=42,
        index=index,
        max_order=max_order,
        approximator="auto",
    )
    x = data[0].reshape(1, -1)
    interaction_values = explainer.explain(x, budget=budget)
    assert interaction_values.index == index
    assert interaction_values.max_order == max_order
    if budget is None:
        budget = 100_000_000_000
    assert interaction_values.estimation_budget <= budget + 2
