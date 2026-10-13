"""The ``TreeExplainer`` front end: its path-dependent and interventional routes, the backends
and the batch API."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.tree import TreeExplainer, TreeModel
from tests.shapiq.markers import skip_if_no_lightgbm


@pytest.mark.parametrize(
    ("model_fixture", "data_fixture", "index", "max_order", "class_index"),
    [
        ("dt_reg_model", "background_reg_data", "k-SII", 2, None),
        ("rf_reg_model", "background_reg_data", "k-SII", 2, None),
        ("rf_clf_model", "background_clf_data", "k-SII", 2, 1),
        ("rf_clf_model", "background_clf_data", "SV", 1, 0),
    ],
)
def test_baseline_and_efficiency(
    request, model_fixture, data_fixture, index, max_order, class_index
):
    """The baseline is the sum of the per-tree empty predictions, and all values of the
    explanation plus the baseline equal the prediction of the explained class."""
    model = request.getfixturevalue(model_fixture)
    x = request.getfixturevalue(data_fixture)[0]
    explainer = TreeExplainer(
        model=model, max_order=max_order, min_order=1, index=index, class_index=class_index
    )
    explanation = explainer.explain(x)
    assert isinstance(explanation, InteractionValues)
    assert explainer.baseline_value == pytest.approx(
        sum(tree.empty_prediction for tree in explainer._trees)
    )
    assert explanation.baseline_value == explainer.baseline_value
    if class_index is None:
        prediction = model.predict(x.reshape(1, -1))[0]
    else:
        prediction = model.predict_proba(x.reshape(1, -1))[0, class_index]
    assert sum(explanation.values) + explanation.baseline_value == pytest.approx(prediction)


def test_hand_tree_matches_shap():
    """Shapley values on a hand-built 3-feature tree equal ``shap.TreeExplainer``'s."""
    values = np.asarray([110, 105, 95, 20, 50, 100, 75, 10, 40]) / 110
    tree_model = TreeModel(
        children_left=np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1]),
        children_right=np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1]),
        children_missing=np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1]),
        features=np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2]),
        thresholds=np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2]),
        node_sample_weight=np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30]),
        values=values,
    )
    x_explain = np.asarray([-1, -0.5, 1, 0])
    expected = {(0,): -0.09263158, (1,): -0.12100478, (2,): 0.02727273, (3,): 0.0}
    for index in ("SV", "SII"):
        explanation = TreeExplainer(
            model=tree_model, max_order=1, min_order=1, index=index
        ).explain(x_explain)
        for key, value in expected.items():
            assert explanation[key] == pytest.approx(value, abs=1e-4)
    with pytest.warns(UserWarning):  # SV with max_order=2 is downgraded with a warning
        _ = TreeExplainer(model=tree_model, max_order=2, min_order=1, index="SV")


@pytest.mark.parametrize("mode", ["pathdependent", "interventional"])
def test_min_order_filters_interactions(rf_reg_model, background_reg_data, mode):
    """``min_order`` restricts the returned interactions on both routes (issue #325); the k-SII
    aggregation reports the empty interaction, which the wrapper filters like any other order."""
    x_explain = background_reg_data[0]
    kwargs = {"reference_dataset": background_reg_data[:20]} if mode == "interventional" else {}

    def TreeExplainer(**explainer_kwargs):
        from shapiq.tree import TreeExplainer as _TreeExplainer

        return _TreeExplainer(mode=mode, backend="shapiq", **kwargs, **explainer_kwargs)

    # min_order >= 2 drops order-1 interactions but keeps the higher orders
    explainer = TreeExplainer(model=rf_reg_model, max_order=3, min_order=2, index="k-SII")
    explanation = explainer.explain(x_explain)
    assert explanation.min_order == 2
    assert explanation.max_order == 3
    assert len(explanation.interaction_lookup) > 0
    assert all(len(interaction) >= 2 for interaction in explanation.interaction_lookup)

    # min_order == 0 still injects the empty interaction at the baseline value
    explainer_zero = TreeExplainer(model=rf_reg_model, max_order=2, min_order=0, index="k-SII")
    explanation_zero = explainer_zero.explain(x_explain)
    assert () in explanation_zero.interaction_lookup
    assert explanation_zero[()] == pytest.approx(explanation_zero.baseline_value)
    assert any(len(interaction) == 1 for interaction in explanation_zero.interaction_lookup)

    # min_order == 1 matches the unfiltered default-min_order=0 explainer with the
    # empty interaction removed (regression guard for the default code path)
    explainer_one = TreeExplainer(model=rf_reg_model, max_order=2, min_order=1, index="k-SII")
    explanation_one = explainer_one.explain(x_explain)
    assert explanation_one.min_order == 1
    assert () not in explanation_one.interaction_lookup
    for interaction in explanation_one.interaction_lookup:
        assert explanation_one[interaction] == pytest.approx(explanation_zero[interaction])


def test_min_order_validation(rf_reg_model):
    """Invalid ``min_order`` configurations must be rejected eagerly."""
    with pytest.raises(ValueError, match="min_order"):
        TreeExplainer(model=rf_reg_model, max_order=2, min_order=3, index="k-SII")
    with pytest.raises(ValueError, match="min_order"):
        TreeExplainer(model=rf_reg_model, max_order=2, min_order=-1, index="k-SII")


# Shapley values recorded from ``shap.TreeExplainer`` (see the git history for the snippets):
# (model fixture, data fixture or None, explained row or point, class_index, values, baseline)
_SHAP_CASES = [
    (
        "xgb_reg_model",
        "background_reg_data",
        0,
        None,
        [-2.555832, 28.50987, 1.7708225, -7.8653603, 10.7955885, -0.1877861, 4.549199],
        None,
    ),
    (
        "xgb_clf_model",
        "background_clf_data",
        1,
        1,
        [-0.00543903, -0.15696308, -0.17532629, -0.24037467, 0.00245022, 0.00986468, -0.01556843],
        None,
    ),
    # the instance whose raw float64 value and float32-cast value route differently
    (
        "xgb_clf_model",
        "background_clf_data",
        0,
        1,
        [-0.00163171, 0.05075389, -0.13064955, -0.4421068, 0.00424677, -0.04832656, -0.01364264],
        None,
    ),
    (
        "rf_reg_model",
        "background_reg_data",
        1,
        None,
        [25.8278293, -77.40235947, 0.0, 21.7067263, -4.85542565, 0.0, 4.91330141],
        -0.713665621534487,
    ),
    (
        "rf_clf_model",
        "background_clf_data",
        1,
        1,
        [-0.00537992, 0.0, -0.08206514, -0.03122057, 0.0025626, 0.03182904, 0.03782473],
        0.32,
    ),
    (
        "lightgbm_clf_model",
        "background_clf_data",
        1,
        1,
        [0.0, 0.0, -0.05747963, -0.20128496, 0.0, 0.0, 0.01560273],
        -1.0862557008895362,
    ),
    (
        "et_clf_model",
        "background_clf_data",
        1,
        1,
        [0.00207427, 0.00949552, -0.00108266, -0.03825587, -0.02694092, 0.0170296, 0.02046364],
        0.34,
    ),
    (
        "et_reg_model",
        "background_reg_data",
        1,
        None,
        [19.28673017, -19.87182634, 0.0, 10.89201698, -9.62498263, 0.35992212, 42.31290091],
        -2.56682283435175,
    ),
    (
        "if_clf_model",
        None,
        np.array([0.125, 0.05]),
        None,
        [-1.40624839, -3.21377854],
        11.953360265689595,
    ),
]


@pytest.mark.parametrize(
    ("model_fixture", "data_fixture", "point", "class_index", "sv_shap", "baseline_shap"),
    _SHAP_CASES,
    ids=[f"{case[0]}-{case[2] if isinstance(case[2], int) else 'point'}" for case in _SHAP_CASES],
)
def test_matches_shap_values(
    request, model_fixture, data_fixture, point, class_index, sv_shap, baseline_shap
):
    """Path-dependent Shapley values and baselines equal those recorded from ``shap`` for
    XGBoost, random forests, LightGBM, extra trees and an isolation forest."""
    model = request.getfixturevalue(model_fixture)
    x = request.getfixturevalue(data_fixture)[point] if data_fixture else point
    explainer = TreeExplainer(model=model, max_order=1, index="SV", class_index=class_index)
    explanation = explainer.explain(x=x)
    assert np.allclose(sv_shap, explanation.get_n_order_values(1), rtol=1e-5)
    if baseline_shap is not None:
        assert explanation.baseline_value == pytest.approx(baseline_shap, rel=1e-4)


@skip_if_no_lightgbm
def test_decision_stumps(background_reg_dataset, background_clf_dataset):
    """Tests weather you can explain a decision stumps with the shapiq implementation."""
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from xgboost import XGBClassifier, XGBRegressor

    models_reg = [
        LGBMRegressor(random_state=42, n_estimators=20, max_depth=1),
        XGBRegressor(random_state=42, n_estimators=20, max_depth=1),
        DecisionTreeRegressor(random_state=42, max_depth=1),
        RandomForestRegressor(random_state=42, n_estimators=20, max_depth=1),
    ]

    models_clf = [
        LGBMClassifier(random_state=42, n_estimators=20, max_depth=1),
        XGBClassifier(random_state=42, n_estimators=20, max_depth=1),
        DecisionTreeClassifier(random_state=42, max_depth=1),
        RandomForestClassifier(random_state=42, n_estimators=20, max_depth=1),
    ]

    for model in models_reg:
        X, y = background_reg_dataset
        model.fit(X, y)

        explainer = TreeExplainer(model=model, max_order=3, index="k-SII")
        x_explain = X[0]
        explanation = explainer.explain(x_explain)

        efficiency = sum(explanation.values)

        pred = model.predict(x_explain.reshape(1, -1))
        assert pred == pytest.approx(efficiency, rel=1e-5)

    for model in models_clf:
        X, y = background_clf_dataset
        model.fit(X, y)

        explainer = TreeExplainer(model=model, max_order=3, index="k-SII", class_index=0)
        x_explain = X[1]
        explanation = explainer.explain(x_explain)

        efficiency = sum(explanation.values)

        if isinstance(model, RandomForestClassifier | DecisionTreeClassifier):
            pred = model.predict_proba(x_explain.reshape(1, -1))[0, 0]
        elif isinstance(model, XGBClassifier):
            pred = model.predict(x_explain.reshape(1, -1), output_margin=True)[0, 0]
        else:  # skip lightgbm
            continue

        assert pred == pytest.approx(efficiency, rel=1e-5)


def test_interventional_missing_reference_dataset_raises(rf_reg_model):
    """``mode='interventional'`` requires a ``reference_dataset`` already at construction."""
    with pytest.raises(ValueError, match="reference_dataset"):
        TreeExplainer(model=rf_reg_model, mode="interventional", max_order=2, index="SII")


@pytest.mark.parametrize(
    ("model_fixture", "data_fixture", "index", "max_order", "class_index"),
    [
        ("dt_reg_model", "background_reg_data", "SII", 2, None),
        ("rf_reg_model", "background_reg_data", "SV", 1, None),
        ("rf_clf_model", "background_clf_data", "SII", 2, 1),
        ("dt_reg_model", "background_reg_data", "k-SII", 3, None),
    ],
)
def test_interventional_route_matches_the_direct_explainer(
    request, model_fixture, data_fixture, index, max_order, class_index
):
    """``mode="interventional"`` wires up ``InterventionalTreeSHAPIQ``: same value on every
    interaction, the mean prediction over the reference rows as baseline, every order up to
    ``max_order`` reported, and (k-SII) values that sum to the prediction."""
    from shapiq.tree import InterventionalTreeSHAPIQ

    model = request.getfixturevalue(model_fixture)
    data = request.getfixturevalue(data_fixture)
    reference, x = data[:20], data[0]
    wrapper = TreeExplainer(
        model=model,
        mode="interventional",
        reference_dataset=reference,
        max_order=max_order,
        min_order=1,
        index=index,
        class_index=class_index,
        backend="shapiq",
    )
    wrapper._init_explainers()
    assert wrapper._interventional_explainer is not None
    assert wrapper._pathdependent_explainer is None

    explanation = wrapper.explain(x)
    assert isinstance(explanation, InteractionValues)
    assert explanation.max_order == max_order
    assert set(range(1, max_order + 1)) <= {len(k) for k in explanation.interaction_lookup}
    assert explanation.min_order == 1
    assert () not in explanation.interaction_lookup

    if class_index is None:
        predictions = model.predict(reference)
        prediction = float(model.predict(x.reshape(1, -1))[0])
    else:
        predictions = model.predict_proba(reference)[:, class_index]
        prediction = float(model.predict_proba(x.reshape(1, -1))[0, class_index])
    assert explanation.baseline_value == pytest.approx(float(predictions.mean()), rel=1e-5)

    if index == "k-SII":  # aggregated on the way out: check efficiency instead of the values
        total = sum(explanation[k] for k in explanation.interaction_lookup if k)
        assert total + explanation.baseline_value == pytest.approx(prediction)
        return
    direct = InterventionalTreeSHAPIQ(
        model=model, data=reference, max_order=max_order, index=index, class_index=class_index
    )
    direct_values = direct.explain_function(x)
    assert wrapper.baseline_value == pytest.approx(direct.baseline_value)
    for interaction in explanation.interaction_lookup:
        if interaction:
            assert explanation[interaction] == pytest.approx(direct_values[interaction], abs=1e-9)


def test_baseline_value_per_mode(rf_reg_model, background_reg_data, monkeypatch):
    """``baseline_value`` is the empty prediction of the mode the explainer runs in.

    Path-dependent: the sum of the per-tree (coverage-weighted) empty predictions.
    Interventional: the mean ensemble prediction over the reference dataset — not the
    path-dependent value, which is what a woodelf-routed explanation used to report.
    """
    from shapiq.tree import InterventionalTreeSHAPIQ

    # lower the cut-off so the 100-row reference routes even explain() to Woodelf
    monkeypatch.setattr("shapiq.tree.explainer._WOODELF_INTERVENTIONAL_CUTOFF", 100)
    reference = background_reg_data  # 100 rows: 100 * 1 >= 100 crosses the patched cut-off
    x_explain = background_reg_data[0]

    pathdependent = TreeExplainer(model=rf_reg_model, max_order=1, index="SV")
    assert pathdependent.baseline_value == pytest.approx(
        sum(tree.empty_prediction for tree in pathdependent._trees)
    )

    interventional = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=0,
        index="SV",
    )
    expected_baseline = InterventionalTreeSHAPIQ(
        model=rf_reg_model, data=reference, max_order=1, index="SV"
    ).baseline_value
    assert interventional.baseline_value == pytest.approx(expected_baseline)
    assert expected_baseline == pytest.approx(float(rf_reg_model.predict(reference).mean()))

    # the woodelf-routed single-instance explanation must carry the interventional baseline
    pytest.importorskip("woodelf")
    assert interventional._should_use_woodelf(1)  # guard: Woodelf, not the fallback
    explanation = interventional.explain(x_explain)
    assert explanation.baseline_value == pytest.approx(expected_baseline)
    assert explanation[()] == pytest.approx(expected_baseline)


def test_woodelf_fallback_warns_without_woodelf(rf_reg_model, background_reg_data, monkeypatch):
    """Without the optional woodelf dependency the explainer warns and falls back to shapiq."""
    import importlib.util
    import warnings

    from shapiq.tree import WoodelfNotAvailableWarning

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        return None if name == "woodelf" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    # 100 background rows cross the (patched) interventional cut-off, so this would route there
    monkeypatch.setattr("shapiq.tree.explainer._WOODELF_INTERVENTIONAL_CUTOFF", 100)
    explainer = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=background_reg_data[:100],
        max_order=1,
        min_order=1,
        index="SV",
    )
    with pytest.warns(WoodelfNotAvailableWarning, match="woodelf"):
        assert not explainer._should_use_woodelf(1)
        explanation = explainer.explain(background_reg_data[0])
    assert type(explanation).__name__ == "InteractionValues"
    assert explainer._interventional_explainer is not None  # the shapiq fallback computed it

    # path-dependent explanations never route to Woodelf: the quadrature default computes
    # every order and index natively, so no cut-off applies and nothing warns
    pathdependent = TreeExplainer(model=rf_reg_model, max_order=2, min_order=1, index="SII")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not pathdependent._should_use_woodelf(1000)
        explanation = pathdependent.explain(background_reg_data[0])
    assert type(explanation).__name__ == "InteractionValues"


@pytest.mark.parametrize("index", ["k-SII", "BV"])
def test_explain_X_pathdependent_without_woodelf(dt_reg_model, background_reg_data, index):
    """``explain_X`` on the default path-dependent route works without woodelf.

    The shapiq branch of ``explain_X`` (lazy init + per-instance quadrature + batch
    wrapping) must produce one row per instance, each matching the single-instance
    ``explain``.
    """
    max_order = 2 if index == "k-SII" else 1
    X_explain = background_reg_data[:4]
    explainer = TreeExplainer(
        model=dt_reg_model, max_order=max_order, min_order=1, index=index, backend="shapiq"
    )
    batch = explainer.explain_X(X_explain)
    assert len(batch) == len(X_explain)
    for row, x in enumerate(X_explain):
        single = explainer.explain(x)
        assert batch[row].index == index
        for interaction in single.interactions:
            assert batch[row][interaction] == pytest.approx(single[interaction], abs=1e-12)


def test_backend_shapiq_forces_shapiq(rf_reg_model, background_reg_data, monkeypatch):
    """``backend='shapiq'`` never routes to Woodelf, even past the cut-offs, and never warns."""
    import warnings

    monkeypatch.setattr("shapiq.tree.explainer._WOODELF_INTERVENTIONAL_CUTOFF", 100)
    reference = background_reg_data[:10]
    x_explain = background_reg_data[0]

    explainer = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=1,
        index="SV",
        backend="shapiq",
    )
    assert not explainer._should_use_woodelf(20)  # 10 * 20 crosses the patched cut-off
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # forcing shapiq is a choice, not something to warn about
        explanation = explainer.explain(x_explain)
    assert type(explanation).__name__ == "InteractionValues"


def test_backend_woodelf_forces_woodelf(dt_reg_model, background_reg_data):
    """``backend='woodelf'`` routes to Woodelf even under the cut-offs and matches shapiq."""
    pytest.importorskip("woodelf")
    x_explain = background_reg_data[0]
    n_features = background_reg_data.shape[1]

    forced = TreeExplainer(
        model=dt_reg_model, max_order=1, min_order=1, index="SV", backend="woodelf"
    )
    assert forced._should_use_woodelf(1)  # a single path-dependent SV instance is under the cut-off
    forced_explanation = forced.explain(x_explain)

    shapiq_explanation = TreeExplainer(
        model=dt_reg_model, max_order=1, min_order=1, index="SV", backend="shapiq"
    ).explain(x_explain)

    for feature in range(n_features):
        assert forced_explanation[(feature,)] == pytest.approx(
            shapiq_explanation[(feature,)], abs=1e-5
        )


def test_woodelf_ksii_matches_shapiq(dt_reg_model, background_reg_data):
    """Woodelf serves ``k-SII`` by computing SII and aggregating, exactly like the shapiq path.

    Both the single-instance ``explain`` and the batched ``explain_X`` of a forced woodelf
    backend must match the forced shapiq backend.
    """
    pytest.importorskip("woodelf")
    X_explain = background_reg_data[:5]

    woodelf = TreeExplainer(
        model=dt_reg_model, max_order=2, min_order=1, index="k-SII", backend="woodelf"
    )
    shapiq_ = TreeExplainer(
        model=dt_reg_model, max_order=2, min_order=1, index="k-SII", backend="shapiq"
    )

    # single instance: identical interactions over the union (missing on one side means zero)
    woodelf_explanation = woodelf.explain(X_explain[0])
    shapiq_explanation = shapiq_.explain(X_explain[0])
    assert woodelf_explanation.index == "k-SII"
    interactions = set(woodelf_explanation.interactions) | set(shapiq_explanation.interactions)
    assert {len(interaction) for interaction in interactions} == {1, 2}
    for interaction in interactions:
        # rel covers float32 woodelf arrays on large values, abs covers exact zeros
        assert woodelf_explanation[interaction] == pytest.approx(
            shapiq_explanation[interaction], rel=1e-5, abs=1e-5
        )

    # batched: the vectorized woodelf-format dicts must match row by row
    woodelf_values = woodelf.explain_X(X_explain).values
    shapiq_values = shapiq_.explain_X(X_explain).values
    zeros = np.zeros(len(X_explain))
    for subset in set(woodelf_values) | set(shapiq_values):
        assert woodelf_values.get(subset, zeros) == pytest.approx(
            shapiq_values.get(subset, zeros), rel=1e-5, abs=1e-5
        )


def test_backend_validation_raises(dt_reg_model, background_reg_data, monkeypatch):
    """Invalid or unusable backend configurations fail at construction time."""
    import importlib.util

    # unknown backend name
    with pytest.raises(ValueError, match="backend='banzhaf-machine'"):
        TreeExplainer(model=dt_reg_model, backend="banzhaf-machine")

    # backend='woodelf' with a configuration Woodelf cannot serve (already-parsed trees);
    # the config check needs no woodelf import, so pretend the package is installed to make
    # this case testable in every environment
    import importlib.machinery

    from shapiq.tree.validation import validate_tree_model

    real_find_spec = importlib.util.find_spec

    def fake_installed(name, *args, **kwargs):
        if name == "woodelf":
            return importlib.machinery.ModuleSpec("woodelf", None)
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_installed)
    with pytest.raises(ValueError, match="original model object"):
        TreeExplainer(model=validate_tree_model(dt_reg_model), backend="woodelf")

    # without the woodelf package, forcing it raises; path-dependent Banzhaf indices do not
    # need it (the quadrature default computes them natively)
    def fake_missing(name, *args, **kwargs):
        return None if name == "woodelf" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_missing)
    with pytest.raises(ImportError, match="woodelf-explainer"):
        TreeExplainer(model=dt_reg_model, backend="woodelf")
    banzhaf = TreeExplainer(model=dt_reg_model, index="BV", backend="auto")
    assert banzhaf.explain(background_reg_data[0]).index == "BV"


def test_woodelf_interventional_matches_direct_explainer(
    rf_reg_model, background_reg_data, monkeypatch
):
    """Background SHAP over 20 rows x 10 background rows routes through Woodelf.

    ``10 * 20 >= 100`` crosses the patched interventional cut-off, so ``explain_X`` no longer
    runs the shapiq kernel. The per-feature values must still match the direct
    :class:`InterventionalTreeSHAPIQ`.
    """
    from shapiq.tree import InterventionalTreeSHAPIQ

    pytest.importorskip("woodelf")
    monkeypatch.setattr("shapiq.tree.explainer._WOODELF_INTERVENTIONAL_CUTOFF", 100)
    reference = background_reg_data[:10]
    x_explain = background_reg_data[10:30]
    n_features = background_reg_data.shape[1]

    explainer = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=1,
        index="SV",
    )
    assert explainer._should_use_woodelf(len(x_explain))  # guard: Woodelf, not the fallback
    woodelf_values = explainer.explain_X(x_explain).values

    direct = InterventionalTreeSHAPIQ(model=rf_reg_model, data=reference, max_order=1, index="SV")
    # Woodelf omits a feature entirely when it is zero for every row, so default to zeros.
    zeros = np.zeros(len(x_explain))
    for row, x in enumerate(x_explain):
        direct_iv = direct.explain_function(x)
        for feature in range(n_features):
            woodelf_value = woodelf_values.get((feature,), zeros)[row]
            assert woodelf_value == pytest.approx(direct_iv[(feature,)], abs=1e-5)


def test_explain_X_returns_interaction_values_batch(rf_reg_model, background_reg_data, monkeypatch):
    """``explain_X`` returns a lazy sequence of ``InteractionValues`` (the base contract).

    ``10 * 20 >= 100`` crosses the patched interventional cut-off, so the batch is computed by
    Woodelf — but the result must still behave like one :class:`InteractionValues` per instance,
    matching the per-instance ``explain`` (which stays under the cut-off and runs the shapiq
    kernel), while exposing the raw vectorized arrays via ``.values``.
    """
    pytest.importorskip("woodelf")
    from shapiq.interaction_values import InteractionValues, InteractionValuesBatch

    monkeypatch.setattr("shapiq.tree.explainer._WOODELF_INTERVENTIONAL_CUTOFF", 100)
    reference = background_reg_data[:10]
    X_explain = background_reg_data[10:30]

    explainer = TreeExplainer(
        model=rf_reg_model,
        mode="interventional",
        reference_dataset=reference,
        max_order=1,
        min_order=1,
        index="SV",
    )
    assert explainer._should_use_woodelf(len(X_explain))  # guard: Woodelf, not the fallback
    explanations = explainer.explain_X(X_explain)

    # sequence protocol: len, indexing (incl. negative), iteration, slicing
    assert isinstance(explanations, InteractionValuesBatch)
    assert len(explanations) == len(X_explain)
    assert isinstance(explanations[0], InteractionValues)
    assert explanations[-1][(0,)] == explanations[len(X_explain) - 1][(0,)]
    assert len(explanations[2:5]) == 3
    assert all(isinstance(explanation, InteractionValues) for explanation in explanations)

    # the raw vectorized arrays are exposed without materialization
    assert all(len(values) == len(X_explain) for values in explanations.values.values())

    # each row matches the single-instance explanation (computed by the shapiq kernel here)
    for row, x in enumerate(X_explain):
        single = explainer.explain(x)
        for feature in range(X_explain.shape[1]):
            assert explanations[row][(feature,)] == pytest.approx(
                single[(feature,)], rel=1e-5, abs=1e-5
            )


@pytest.fixture
def deep_dt_reg_model(background_reg_dataset):
    """A decision tree deep enough that its Moebius sets exceed ``max_order=3``.

    The shared depth-3 fixtures cannot tell the interaction indices apart: when no Moebius set
    is larger than ``max_order``, STII, FSII and FBII all reduce to the plain Moebius transform
    and become numerically identical. Depth 6 separates them while staying under the depth-16
    limit of Woodelf's path-dependent kernel.
    """
    from sklearn.tree import DecisionTreeRegressor

    X, y = background_reg_dataset
    return DecisionTreeRegressor(random_state=42, max_depth=6).fit(X, y)


_WOODELF_INTERVENTIONAL_INDICES = ["SII", "BII", "STII", "FSII", "FBII"]
# As the shapiq backend does not support path-dependent STII, FSII and FBII,
# only SII and BII are tested for path-dependent mode.
_WOODELF_PATHDEPENDENT_INDICES = ["SII", "BII"]


def _assert_woodelf_matches_shapiq(
    woodelf_explainer, shapiq_explainer, x_explain, index, min_order, max_order
):
    """Assert the two explainers explain ``x_explain`` the same way."""
    assert woodelf_explainer._should_use_woodelf(1)  # guard: Woodelf on one side ...
    assert not shapiq_explainer._should_use_woodelf(1)  # ... and the shapiq kernel on the other

    explanation = woodelf_explainer.explain(x_explain)
    reference_explanation = shapiq_explainer.explain(x_explain)

    assert explanation.index == index
    assert explanation.min_order == min_order
    assert explanation.max_order == max_order
    assert explanation[()] == pytest.approx(woodelf_explainer.baseline_value)

    # a subset missing on one side is an exact zero there, so compare over the union
    interactions = set(explanation.interactions) | set(reference_explanation.interactions)
    assert {len(interaction) for interaction in interactions} == set(range(max_order + 1))
    for interaction in interactions:
        if interaction:  # skip the empty interaction, checked against the baseline above
            assert explanation[interaction] == pytest.approx(
                reference_explanation[interaction], abs=1e-5
            )


@pytest.mark.parametrize("index", _WOODELF_PATHDEPENDENT_INDICES)
def test_woodelf_pathdependent_matches_shapiq(deep_dt_reg_model, background_reg_data, index):
    """Forced path-dependent Woodelf with ``min_order=0``, ``max_order=3`` matches shapiq."""
    pytest.importorskip("woodelf")
    woodelf_explainer = TreeExplainer(
        model=deep_dt_reg_model, max_order=3, min_order=0, index=index, backend="woodelf"
    )
    shap_explainer = TreeExplainer(
        model=deep_dt_reg_model, max_order=3, min_order=0, index=index, backend="shapiq"
    )
    _assert_woodelf_matches_shapiq(
        woodelf_explainer,
        shap_explainer,
        background_reg_data[0],
        index=index,
        min_order=0,
        max_order=3,
    )


@pytest.mark.parametrize("index", _WOODELF_INTERVENTIONAL_INDICES)
def test_woodelf_interventional_matches_shapiq(deep_dt_reg_model, background_reg_data, index):
    """Forced interventional Woodelf with ``min_order=0``, ``max_order=3`` matches shapiq."""
    pytest.importorskip("woodelf")
    reference_dataset = background_reg_data[:10]
    woodelf_explainer = TreeExplainer(
        model=deep_dt_reg_model,
        mode="interventional",
        reference_dataset=reference_dataset,
        max_order=3,
        min_order=0,
        index=index,
        backend="woodelf",
    )
    shap_explainer = TreeExplainer(
        model=deep_dt_reg_model,
        mode="interventional",
        reference_dataset=reference_dataset,
        max_order=3,
        min_order=0,
        index=index,
        backend="shapiq",
    )
    _assert_woodelf_matches_shapiq(
        woodelf_explainer,
        shap_explainer,
        background_reg_data[10],
        index=index,
        min_order=0,
        max_order=3,
    )


@pytest.mark.parametrize(
    ("model_fixture", "data_fixture", "n_classes"),
    [
        ("rf_clf_model", "background_clf_dataset", 3),
        ("rf_clf_binary_model", "background_clf_dataset_binary", 2),
        ("xgb_clf_model", "background_clf_dataset", 3),
    ],
    ids=["rf-multiclass", "rf-binary", "xgb-multiclass"],
)
def test_woodelf_pathdependent_honors_class_index(
    model_fixture,
    data_fixture,
    n_classes,
    request,
):
    """Woodelf must explain the class ``class_index`` asks for.

    ``backend='woodelf'`` forces every class here to be computed by Woodelf rather than
    the shapiq kernel. Each one must match :class:`~shapiq.tree.TreeSHAPIQ` run directly
    on the class-selected trees. Sklearn ensembles select a class by slicing leaf values
    and boosters by filtering trees, so both mechanisms are covered.
    """
    from shapiq.tree import TreeSHAPIQ
    from shapiq.tree.validation import validate_tree_model

    pytest.importorskip("woodelf")
    model = request.getfixturevalue(model_fixture)
    X, _ = request.getfixturevalue(data_fixture)
    x_explain = np.asarray(X)[2]

    for class_index in range(n_classes):
        explainer = TreeExplainer(
            model=model,
            mode="pathdependent",
            max_order=2,
            min_order=1,
            index="SII",
            class_index=class_index,
            backend="woodelf",
        )
        assert explainer._should_use_woodelf(1)  # guard: Woodelf, not the fallback
        explanation = explainer.explain(x_explain)

        per_tree = [
            TreeSHAPIQ(model=tree, max_order=2, index="SII").explain(x_explain)
            for tree in validate_tree_model(model, class_label=class_index)
        ]
        reference = sum(per_tree[1:], start=per_tree[0])

        interactions = set(explanation.interactions) | set(reference.interactions)
        for interaction in interactions:
            if (
                len(interaction) >= 1
            ):  # the empty interaction carries the baseline, not a class value
                assert explanation[interaction] == pytest.approx(reference[interaction], abs=1e-5)


def test_tree_model_batch_predict(rf_reg_model, background_reg_data):
    """``TreeModel.predict`` is the batch companion of ``predict_one``.

    Per tree it must equal the per-row ``predict_one`` results, and summed over the validated
    ensemble it must reproduce the original model's predictions.
    """
    from shapiq.tree.base import predict_ensemble
    from shapiq.tree.validation import validate_tree_model

    X = background_reg_data[:15]
    trees = validate_tree_model(rf_reg_model)

    for tree in trees:
        batched = tree.predict(X)
        assert batched.shape == (len(X),)
        assert batched == pytest.approx([tree.predict_one(x) for x in X])

    assert predict_ensemble(trees, X) == pytest.approx(rf_reg_model.predict(X), rel=1e-5)
