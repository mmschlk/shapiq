"""This test module contains all tests regarding the interaction explainer for the shapiq package."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from shapiq import InteractionValues
from shapiq.approximator import RegressionFSII
from shapiq.explainer.tabular import (
    TabularExplainer,
    TabularExplainerApproximators,
    TabularExplainerImputers,
    TabularExplainerIndices,
)
from tests.fixtures.data import BUDGET_NR_FEATURES
from tests.utils import get_expected_index_or_skip

MAX_ORDERS = [2, 3]


@pytest.mark.parametrize("index", get_args(TabularExplainerIndices))
@pytest.mark.parametrize("max_order", [1, 2, 3])
# @pytest.mark.parametrize("imputer", get_args(TabularExplainerImputers))
def test_init_params(dt_reg_model, background_reg_data, index, max_order):
    """Test the initialization of the interaction explainer."""
    expected_index = get_expected_index_or_skip(index, max_order)
    model_function = dt_reg_model.predict
    explainer = TabularExplainer(
        model=model_function,
        data=background_reg_data,
        random_state=42,
        index=index,
        max_order=max_order,
        approximator="auto",
    )
    assert explainer.index == expected_index
    assert explainer.max_order == max_order
    # test defaults
    if max_order == 1 and explainer.index == "SV":
        assert explainer.approximator.__class__.__name__ == "KernelSHAP"
    elif max_order == 1 and explainer.index == "BV":
        assert explainer.approximator.__class__.__name__ == "RegressionFBII"
        assert explainer.approximator.max_order == 1
    elif index == "FSII":
        assert explainer.approximator.__class__.__name__ == "RegressionFSII"
    elif index == "FBII":
        assert explainer.approximator.__class__.__name__ == "RegressionFBII"
    elif index in ("SII", "k-SII"):
        assert explainer.approximator.__class__.__name__ == "KernelSHAPIQ"
    else:
        assert explainer.approximator.__class__.__name__ == "SVARMIQ"


def test_auto_params(dt_reg_model, background_reg_data):
    """Test the initialization of the interaction explainer."""
    model_function = dt_reg_model.predict
    explainer = TabularExplainer(
        model=model_function,
        data=background_reg_data,
    )
    assert explainer.index == "k-SII"
    assert explainer.approximator.index == "k-SII"
    assert explainer.max_order == 2
    assert explainer.approximator.__class__.__name__ == "KernelSHAPIQ"


def test_init_params_error_and_warning(dt_reg_model, background_reg_data):
    """Test the initialization of the interaction explainer."""
    model_function = dt_reg_model.predict
    with pytest.raises(ValueError):
        TabularExplainer(
            model=model_function,
            data=background_reg_data,
            index="invalid",
            max_order=0,
        )
    with pytest.warns():
        TabularExplainer(
            model=model_function,
            data=background_reg_data,
            max_order=1,
            index="k-SII",  # not SV and order is 1
        )
    with pytest.warns():
        TabularExplainer(
            model=model_function,
            data=background_reg_data,
            max_order=1,
            index="FBII",  # not BV and order is 1
        )
    with pytest.warns():
        TabularExplainer(
            model=model_function,
            data=background_reg_data,
            index="SV",
            max_order=2,  # higher than 1 and index is SV or BV
        )
    with pytest.warns():
        TabularExplainer(
            model=model_function,
            data=background_reg_data,
            index="BV",
            max_order=2,  # higher than 1 and index is SV or BV
        )


def test_init_params_approx(dt_reg_model, background_reg_data):
    """Test the initialization of the tabular explainer."""
    data = background_reg_data
    model_function = dt_reg_model.predict
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
    assert explainer.approximator.__class__.__name__ == "RegressionFSII"

    # init explainer with manual approximator
    approximator = RegressionFSII(n=9, max_order=2)
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        approximator=approximator,
    )
    assert explainer.approximator.__class__.__name__ == "RegressionFSII"
    assert explainer.approximator == approximator


@pytest.mark.parametrize("approximator", get_args(TabularExplainerApproximators))
@pytest.mark.parametrize("max_order", [*MAX_ORDERS, 1])
def test_init_params_approx_params(dt_reg_model, background_reg_data, approximator, max_order):
    """Test the initialization of the tabular explainer."""
    explainer = TabularExplainer(
        approximator=approximator,
        model=dt_reg_model,
        data=background_reg_data,
        max_order=max_order,
    )
    if approximator == "spex":
        pytest.skip("Spex works only for larger datasets/budgets.")
    iv = explainer.explain(background_reg_data[0], budget=BUDGET_NR_FEATURES)
    assert isinstance(iv, InteractionValues)


BUDGETS = [2**5, 2**8, BUDGET_NR_FEATURES]


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("max_order", MAX_ORDERS)
@pytest.mark.parametrize("imputer", get_args(TabularExplainerImputers))
def test_explain(dt_reg_model, background_reg_data, budget, max_order, imputer):
    """Test the initialization of the interaction explainer."""
    index = "FSII"
    _ = get_expected_index_or_skip(index, max_order)

    model_function = dt_reg_model.predict
    data = background_reg_data
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
    assert interaction_values.estimation_budget <= budget + 2
    interaction_values0 = explainer.explain(x, budget=budget, random_state=0)
    interaction_values2 = explainer.explain(x, budget=budget, random_state=0)
    assert np.allclose(
        interaction_values0.get_n_order_values(1),
        interaction_values2.get_n_order_values(1),
    )
    assert np.allclose(
        interaction_values0.get_n_order_values(2),
        interaction_values2.get_n_order_values(2),
    )

    # test for efficiency
    prediction = float(model_function(x)[0])
    sum_of_values = float(np.sum(interaction_values.values))
    assert pytest.approx(interaction_values[()]) == interaction_values.baseline_value
    assert pytest.approx(sum_of_values, 0.01) == prediction


def test_against_shap_linear():
    """Tests weather TabularExplainer yields similar results as SHAP with a basic linear model."""
    n_samples = 3
    dim = 5
    rng = np.random.default_rng(42)

    def make_linear_model():
        w = rng.normal(size=dim)

        def model(X: np.ndarray):
            return np.dot(X, w)

        return model

    X = rng.normal(size=(n_samples, dim))
    model = make_linear_model()
    # The following code is commented out because it requires SHAP to be installed.
    """
    # import shap
    # compute with shap
    # explainer_shap = shap.explainers.Exact(model, X)
    # shap_values = explainer_shap(X).values
    # print(shap_values)
    """
    shap_values = np.array(
        [
            [-0.29565839, -0.36698085, -0.55970434, 0.22567077, 0.05852208],
            [1.08513574, 0.06365536, 0.46312977, -0.61532757, 0.00370387],
            [-0.78947735, 0.30332549, 0.09657457, 0.38965679, -0.06222595],
        ],
    )

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
    shapiq_values = explainer_shapiq.explain_X(X, budget=2**dim)
    shapiq_values = np.array([values.get_n_order_values(1) for values in shapiq_values])

    assert np.allclose(shap_values, shapiq_values, atol=1e-5)


def test_explain_X_progressbar():
    """Tests if the progress bar is shown when verbose is set to True."""
    n_samples = 3
    dim = 5
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, dim))

    def model(X: np.ndarray):
        return np.dot(X, np.ones(dim))

    explainer = TabularExplainer(
        model=model,
        data=X,
        random_state=42,
        index="SV",
        max_order=1,
    )
    _ = explainer.explain_X(X, budget=2**dim, verbose=True)
    _ = explainer.explain_X(X, budget=2**dim, verbose=False)


@pytest.mark.parametrize("approximator", get_args(TabularExplainerApproximators))
def test_explain_sv(dt_reg_model, background_reg_data, approximator):
    """Tests if init and compute works for SV for different estimators."""
    model_function = dt_reg_model.predict
    data = background_reg_data
    explainer = TabularExplainer(
        model=model_function,
        data=data,
        random_state=42,
        index="SV",
        max_order=1,
        approximator=approximator,
    )
    x = data[0].reshape(1, -1)
    if approximator == "spex":
        pytest.skip("Spex works only for larger datasets/budgets.")
    interaction_values = explainer.explain(x, budget=20)
    assert interaction_values.index == "SV"
    assert interaction_values.max_order == 1
