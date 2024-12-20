"""This test module contains all tests regarding the interaction explainer for the shapiq package."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from shapiq.approximator import RegressionFSII
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


INDICES = ["SII", "k-SII", "STII", "FSII"]
MAX_ORDERS = [2, 3]
IMPUTER = ["marginal", "conditional", "baseline"]
APPROXIMATOR = ["regression", "montecarlo", "permutation"]


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
        assert explainer._approximator.__class__.__name__ == "SVARMIQ"


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


def test_init_params_error_and_warning(dt_model, data):
    """Test the initialization of the interaction explainer."""
    model_function = dt_model.predict
    with pytest.raises(ValueError):
        TabularExplainer(model=model_function, data=data, index="invalid", max_order=0)
    with pytest.warns():
        TabularExplainer(
            model=model_function,
            data=data,
            max_order=1,
        )
    with pytest.warns():
        TabularExplainer(
            model=model_function,
            data=data,
            index="SV",
        )


def test_init_params_approx(dt_model, data):
    """Test the initialization of the tabular explainer."""
    model_function = dt_model.predict
    with pytest.raises(ValueError):
        TabularExplainer(
            model=model_function,
            data=data,
            approximator="invalid",
        )
    explainer = TabularExplainer(
        approximator="regression",
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


@pytest.mark.parametrize("approximator", APPROXIMATOR)
@pytest.mark.parametrize("max_order", MAX_ORDERS + [1])
def test_init_params_approx_params(dt_model, data, approximator, max_order):
    """Test the initialization of the tabular explainer."""
    explainer = TabularExplainer(
        approximator=approximator, model=dt_model, data=data, max_order=max_order
    )
    iv = explainer.explain(data[0])
    assert iv.__class__.__name__ == "InteractionValues"


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
        interaction_values0.get_n_order_values(1), interaction_values2.get_n_order_values(1)
    )
    assert np.allclose(
        interaction_values0.get_n_order_values(2), interaction_values2.get_n_order_values(2)
    )

    # test for efficiency
    if index in ("FSII", "k-SII"):
        prediction = float(model_function(x)[0])
        sum_of_values = float(np.sum(interaction_values.values) + interaction_values.baseline_value)
        assert interaction_values[()] == 0.0
        assert pytest.approx(sum_of_values, 0.01) == prediction


def test_against_shap_linear():
    """Tests weather TabularExplainer yields similar results as SHAP with a basic linear model."""
    import shap

    n_samples = 3
    dim = 5

    def make_linear_model():
        w = np.random.default_rng().normal(size=dim)

        def model(X: np.ndarray):
            return np.dot(X, w)

        return model

    X = np.random.default_rng().normal(size=(n_samples, dim))
    model = make_linear_model()

    # compute with shap
    explainer_shap = shap.explainers.Exact(model, X)
    shap_values = explainer_shap(X).values

    # compute with shapiq
    explainer_shapiq = TabularExplainer(
        model=model,
        data=X,
        random_state=42,
        index="SV",
        max_order=1,
        approximator="auto",
        imputer="marginal",
    )
    shapiq_values = explainer_shapiq.explain_X(X)
    shapiq_values = np.array([values.get_n_order_values(1) for values in shapiq_values])

    assert np.allclose(shap_values, shapiq_values, atol=1e-5)
