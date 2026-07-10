"""Correctness tests for the XGBoost and LightGBM tree converters.

The booster natives cannot share a process with torch on this platform:
once torch's native machinery has run, xgboost and lightgbm segfault in
their data-ingestion paths (macOS wheel cohabitation, reproduced outside
any sandbox; OMP_NUM_THREADS=1 saves xgboost but not lightgbm). The
substantive tests therefore run in a fresh interpreter, driven by one
wrapper test; in that process the whole file passes in a few seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import combinations, product

import jax.numpy as jnp
import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
lightgbm = pytest.importorskip("lightgbm")

# the lightgbm sklearn wrapper warns when predicting from plain arrays
pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

FRESH_ENV = "SHAPIQ_BOOSTER_TESTS"
fresh_process = pytest.mark.skipif(
    os.environ.get(FRESH_ENV) != "fresh",
    reason="booster natives segfault in torch-warmed processes; "
    "covered through test_conversions_run_in_a_fresh_process",
)


@pytest.mark.skipif(
    os.environ.get(FRESH_ENV) == "fresh",
    reason="the fresh process runs the substantive tests itself",
)
def test_conversions_run_in_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        env=os.environ | {FRESH_ENV: "fresh"},
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]

from shapiq import (  # noqa: E402
    SII,
    SV,
    DenseCoalitionArray,
    ExactExplainer,
    InterventionalTreeGame,
    TreeExplainer,
    to_tree_model,
)
from shapiq.trees import _lightgbm, _xgboost  # noqa: E402

N_PLAYERS = 5


def test_the_conversion_kernels_are_built():
    # dev and CI environments must compile the kernels; pure-python installs
    # fall back silently, but here a missing build is a broken build
    assert _xgboost._kernel_parse is not None
    assert _lightgbm._kernel_parse is not None


@pytest.fixture(params=["kernel", "python"])
def parse_path(request, monkeypatch):
    """Run a test through the compiled parsers and the pure-Python fallback."""
    if request.param == "python":
        monkeypatch.setattr(_xgboost, "_kernel_parse", None)
        monkeypatch.setattr(_lightgbm, "_kernel_parse", None)
    return request.param
RNG = np.random.default_rng(0)
FEATURES = RNG.normal(size=(400, N_PLAYERS))
TARGET = (
    FEATURES[:, 0] * FEATURES[:, 1] + FEATURES[:, 2] - 0.5 * FEATURES[:, 3] * FEATURES[:, 4]
)


def assert_full_sweep_parity(predict_margin, trees, inputs, baseline, atol=1e-4):
    """Assert the game equals native margins on every mixed point."""
    game = InterventionalTreeGame(trees, inputs=inputs, baseline=baseline)
    masks = list(product([False, True], repeat=N_PLAYERS))
    values = np.asarray(game(DenseCoalitionArray(jnp.asarray(masks))))
    mixed = np.where(np.asarray(masks)[:, :], inputs, baseline)
    native = np.asarray(predict_margin(mixed))
    assert values.shape == native.shape
    assert np.allclose(values, native, atol=atol), np.abs(values - native).max()


@fresh_process
def test_xgboost_regressor_matches_native_margins_everywhere(parse_path):
    model = xgboost.XGBRegressor(n_estimators=30, max_depth=3, base_score=0.6)
    model.fit(FEATURES, TARGET)
    assert_full_sweep_parity(
        model.predict, to_tree_model(model), FEATURES[0], FEATURES.mean(axis=0)
    )


@fresh_process
def test_xgboost_points_on_a_threshold_route_like_xgboost(parse_path):
    # xgboost routes left on x < t, the unified layout on x <= t; the one-ulp
    # threshold shift must make a point sitting exactly on t route right
    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3).fit(FEATURES, TARGET)
    dump = json.loads(model.get_booster().save_raw(raw_format="json"))
    tree = dump["learner"]["gradient_booster"]["model"]["trees"][0]
    feature = int(tree["split_indices"][0])
    threshold = float(tree["split_conditions"][0])
    inputs = FEATURES[0].copy()
    inputs[feature] = threshold  # exactly on the split
    baseline = FEATURES.mean(axis=0)
    baseline[feature] = threshold - 1.0
    assert_full_sweep_parity(model.predict, to_tree_model(model), inputs, baseline)


@fresh_process
def test_xgboost_binary_classifier_explains_the_margin(parse_path):
    labels = (TARGET > 0).astype(int)
    model = xgboost.XGBClassifier(n_estimators=12, max_depth=3).fit(FEATURES, labels)
    assert_full_sweep_parity(
        lambda x: model.predict(x, output_margin=True),
        to_tree_model(model),
        FEATURES[0],
        FEATURES.mean(axis=0),
    )


@fresh_process
def test_xgboost_multiclass_becomes_a_vector_valued_game(parse_path):
    labels = np.digitize(TARGET, np.quantile(TARGET, [0.33, 0.66]))
    model = xgboost.XGBClassifier(n_estimators=9, max_depth=2).fit(FEATURES, labels)
    trees = to_tree_model(model)
    game = InterventionalTreeGame(trees, inputs=FEATURES[0], baseline=FEATURES.mean(axis=0))
    assert game.value_shape == (3,)
    assert_full_sweep_parity(
        lambda x: model.predict(x, output_margin=True),
        trees,
        FEATURES[0],
        FEATURES.mean(axis=0),
    )
    # vector-valued closed form: Shapley values are efficient per class
    explanation = TreeExplainer(game, SV()).explain()
    ends = game(DenseCoalitionArray(jnp.asarray([[False] * N_PLAYERS, [True] * N_PLAYERS])))
    total = sum(np.asarray(explanation((player,))) for player in range(N_PLAYERS))
    assert np.allclose(total, np.asarray(ends[1] - ends[0]), atol=1e-4)


@fresh_process
def test_lightgbm_regressor_matches_native_margins_everywhere(parse_path):
    model = lightgbm.LGBMRegressor(n_estimators=25, max_depth=3, verbose=-1)
    model.fit(FEATURES, TARGET)
    assert_full_sweep_parity(
        model.predict, to_tree_model(model), FEATURES[0], FEATURES.mean(axis=0)
    )


@fresh_process
def test_lightgbm_points_on_a_threshold_route_left_like_lightgbm(parse_path):
    model = lightgbm.LGBMRegressor(n_estimators=8, max_depth=3, verbose=-1)
    model.fit(FEATURES, TARGET)
    root = model.booster_.dump_model()["tree_info"][0]["tree_structure"]
    inputs = FEATURES[0].copy()
    inputs[int(root["split_feature"])] = float(root["threshold"])  # exactly on the split
    assert_full_sweep_parity(
        model.predict, to_tree_model(model), inputs, FEATURES.mean(axis=0)
    )


@fresh_process
def test_lightgbm_multiclass_becomes_a_vector_valued_game(parse_path):
    labels = np.digitize(TARGET, np.quantile(TARGET, [0.33, 0.66]))
    model = lightgbm.LGBMClassifier(n_estimators=9, max_depth=2, verbose=-1)
    model.fit(FEATURES, labels)
    trees = to_tree_model(model)
    game = InterventionalTreeGame(trees, inputs=FEATURES[0], baseline=FEATURES.mean(axis=0))
    assert game.value_shape == (3,)
    assert_full_sweep_parity(
        lambda x: model.predict(x, raw_score=True),
        trees,
        FEATURES[0],
        FEATURES.mean(axis=0),
    )


@fresh_process
def test_tree_explainer_serves_converted_boosters():
    model = xgboost.XGBRegressor(n_estimators=15, max_depth=3).fit(FEATURES, TARGET)
    game = InterventionalTreeGame(
        to_tree_model(model), inputs=FEATURES[0], baseline=FEATURES.mean(axis=0)
    )
    closed_form = TreeExplainer(game, SII(order=2)).explain()
    exact = ExactExplainer(game, SII(order=2)).explain()
    for size in (1, 2):
        for interaction in combinations(range(N_PLAYERS), size):
            assert jnp.allclose(closed_form(interaction), exact(interaction), atol=1e-4)


@fresh_process
def test_kernel_and_python_parses_agree_exactly():
    model = xgboost.XGBRegressor(n_estimators=10, max_depth=3).fit(FEATURES, TARGET)
    kernel_trees, kernel_info = _xgboost._arrays_from_kernel(model.get_booster())
    json_trees, json_info = _xgboost._arrays_from_json(model.get_booster())
    assert kernel_info == json_info
    assert len(kernel_trees) == len(json_trees)
    for kernel_arrays, json_arrays in zip(kernel_trees, json_trees, strict=True):
        for kernel_column, json_column in zip(kernel_arrays, json_arrays, strict=True):
            assert np.allclose(kernel_column, json_column, rtol=1e-7)

    booster = lightgbm.LGBMRegressor(n_estimators=8, max_depth=3, verbose=-1)
    booster.fit(FEATURES, TARGET)
    kernel_trees, kernel_iteration = _lightgbm._arrays_from_kernel(booster.booster_)
    dump_trees, dump_iteration = _lightgbm._arrays_from_dump(booster.booster_)
    assert kernel_iteration == dump_iteration
    assert len(kernel_trees) == len(dump_trees)
    # node orders differ (layout vs preorder); the games must still agree
    game_masks = DenseCoalitionArray(jnp.asarray(list(product([False, True], repeat=N_PLAYERS))))
    kernel_game = InterventionalTreeGame(
        tuple(_lightgbm._tree_from_arrays(a, 0, 1) for a in kernel_trees),
        inputs=FEATURES[0],
        baseline=FEATURES.mean(axis=0),
    )
    dump_game = InterventionalTreeGame(
        tuple(_lightgbm._tree_from_arrays(a, 0, 1) for a in dump_trees),
        inputs=FEATURES[0],
        baseline=FEATURES.mean(axis=0),
    )
    assert jnp.allclose(kernel_game(game_masks), dump_game(game_masks), atol=1e-6)


@fresh_process
def test_catboost_regressor_matches_native_raw_scores_everywhere():
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostRegressor(iterations=25, depth=3, verbose=0)
    model.fit(FEATURES, TARGET)
    assert_full_sweep_parity(
        model.predict, to_tree_model(model), FEATURES[0], FEATURES.mean(axis=0)
    )


@fresh_process
def test_catboost_points_on_a_border_route_like_catboost():
    # catboost sends x > border right; the layout's x <= border is the exact
    # complement, so a point sitting on the border must route left unshifted
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostRegressor(iterations=10, depth=3, verbose=0)
    model.fit(FEATURES, TARGET)
    trees = to_tree_model(model)
    feature = int(trees[0].features[0])
    border = float(trees[0].thresholds[0])
    inputs = FEATURES[0].copy()
    inputs[feature] = border  # exactly on the split
    assert_full_sweep_parity(model.predict, trees, inputs, FEATURES.mean(axis=0))


@fresh_process
def test_catboost_multiclass_becomes_a_vector_valued_game():
    catboost = pytest.importorskip("catboost")
    labels = np.digitize(TARGET, np.quantile(TARGET, [0.33, 0.66]))
    model = catboost.CatBoostClassifier(
        iterations=9, depth=2, verbose=0, loss_function="MultiClass"
    )
    model.fit(FEATURES, labels)
    trees = to_tree_model(model)
    game = InterventionalTreeGame(trees, inputs=FEATURES[0], baseline=FEATURES.mean(axis=0))
    assert game.value_shape == (3,)
    assert_full_sweep_parity(
        lambda x: np.asarray(model.predict(x, prediction_type="RawFormulaVal")),
        trees,
        FEATURES[0],
        FEATURES.mean(axis=0),
    )


@fresh_process
def test_catboost_closed_form_matches_the_exact_explainer():
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostRegressor(iterations=12, depth=3, verbose=0)
    model.fit(FEATURES, TARGET)
    game = InterventionalTreeGame(
        to_tree_model(model), inputs=FEATURES[0], baseline=FEATURES.mean(axis=0)
    )
    closed_form = TreeExplainer(game, SII(order=2)).explain()
    exact = ExactExplainer(game, SII(order=2)).explain()
    for size in (1, 2):
        for interaction in combinations(range(N_PLAYERS), size):
            assert jnp.allclose(closed_form(interaction), exact(interaction), atol=1e-4)


def test_importing_shapiq_never_imports_the_boosting_libraries():
    probe = """
import sys
import numpy as np
from shapiq import TreeModel, to_tree_model

tree = TreeModel(
    children_left=[-1], children_right=[-1], features=[-2],
    thresholds=[np.nan], values=[1.0],
)
to_tree_model(tree)
for library in ("xgboost", "lightgbm", "catboost", "torch", "sklearn"):
    assert library not in sys.modules, f"{library} must stay unimported"
print("lazy")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "lazy" in result.stdout
