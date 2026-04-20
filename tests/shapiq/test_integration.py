"""Integration tests for the public shapiq user flows.

These tests mirror the canonical flows documented in ``README.md`` and
``docs/source/introduction/start.rst`` — load data, fit a real sklearn
model, build a shapiq explainer, call ``.explain()``, then consume the
``InteractionValues`` downstream (serialisation, plots). Their role is to
catch regressions in the seams between sklearn, the imputer, the
approximator, ``InteractionValues`` and the plot module — a class of bug
that the per-module unit tests can miss.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

import shapiq

from .conftest import assert_iv_close


@pytest.fixture(scope="session")
def california_rf():
    """Real sklearn RF on subsampled California housing — matches README."""
    x_data, y_data = shapiq.load_california_housing(to_numpy=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_data), size=150, replace=False)
    x_data, y_data = x_data[idx], y_data[idx]
    model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42, n_jobs=1)
    model.fit(x_data, y_data)
    return model, x_data, y_data


@pytest.mark.parametrize(
    ("index", "max_order"),
    [
        ("SV", 1),
        ("k-SII", 2),
        ("FSII", 2),
        ("STII", 3),
    ],
)
def test_tabular_explainer_readme_flow(california_rf, index, max_order):
    """README quickstart through TabularExplainer, parametrised across indices.

    Asserts the **efficiency axiom** that all four indices satisfy by design:
    summing the interaction values across all subsets up to ``max_order`` --
    ``InteractionValues.values`` includes the empty-coalition entry storing
    the baseline -- recovers the model prediction on ``x``.
    """
    model, x_data, _ = california_rf
    explainer = shapiq.TabularExplainer(
        model=model,
        data=x_data,
        index=index,
        max_order=max_order,
        random_state=42,
    )
    iv = explainer.explain(x_data[0], budget=256)

    assert isinstance(iv, shapiq.InteractionValues)
    assert iv.index == index
    assert iv.max_order == max_order
    assert iv.n_players == x_data.shape[1]
    assert np.all(np.isfinite(iv.values))

    pred = float(model.predict(x_data[:1])[0])
    assert iv.values.sum() == pytest.approx(pred, abs=1e-4)


@pytest.mark.parametrize(
    ("index", "max_order"),
    [
        ("SV", 1),
        ("k-SII", 2),
    ],
)
def test_tree_explainer_efficiency(california_rf, index, max_order):
    """TreeExplainer pointwise efficiency — holds exactly for tree models."""
    model, x_data, _ = california_rf
    x = x_data[0]
    iv = shapiq.TreeExplainer(model=model, index=index, max_order=max_order).explain(x)

    pred = float(model.predict(x.reshape(1, -1))[0])
    assert iv.values.sum() == pytest.approx(pred, abs=1e-4)


def test_agnostic_explainer_on_soum(soum_7, exact_soum_7):
    """AgnosticExplainer on a Game — researcher flow against exact ground truth."""
    iv = shapiq.AgnosticExplainer(game=soum_7, index="k-SII", max_order=2, random_state=42).explain(
        budget=2**7
    )

    ground_truth = exact_soum_7("k-SII", order=2)
    assert_iv_close(iv, ground_truth, atol=1e-6)


def test_interaction_values_roundtrip_and_plots(california_rf, tmp_path):
    """End-to-end consumption: serialise, reload, plot across all plot surfaces."""
    model, x_data, _ = california_rf

    iv = shapiq.TabularExplainer(
        model=model, data=x_data, index="k-SII", max_order=2, random_state=42
    ).explain(x_data[0], budget=256)

    path = tmp_path / "iv.json"
    iv.save(path)
    iv_loaded = shapiq.InteractionValues.load(path)
    assert np.allclose(iv.values, iv_loaded.values)
    assert iv.index == iv_loaded.index
    assert iv.max_order == iv_loaded.max_order

    feature_names = [f"f{i}" for i in range(x_data.shape[1])]
    assert shapiq.network_plot(iv, feature_names=feature_names) is not None
    assert shapiq.stacked_bar_plot(iv, feature_names=feature_names) is not None
    assert shapiq.bar_plot([iv], feature_names=feature_names) is not None
    assert shapiq.force_plot(iv.get_n_order(order=1), feature_names=feature_names) is not None
    assert shapiq.upset_plot(iv, feature_names=feature_names) is not None
    plt.close("all")
