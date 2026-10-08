"""The chain of trust of the ground truth.

1. Closed-form values validate the brute-force computer.
2. Brute force validates every structured computer on small games, for every index and order the
   structured computer claims to support (and brute force can compute).

Only after this are the structured computers used as ground truth for large games.
"""

from __future__ import annotations

import importlib
import importlib.util
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq_benchmark.computers import (
    BruteForceComputer,
    Computer,
    InterventionalTreeComputer,
    MoebiusComputer,
    NearestNeighborComputer,
    PathDependentTreeComputer,
    ProductKernelComputer,
)
from shapiq_games import (
    SOUM,
    DummyGame,
    InterventionalTreeGame,
    KNNGame,
    PathDependentTreeGame,
    ProductKernelGame,
    ThresholdNNGame,
    UnanimityGame,
    WeightedKNNGame,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.mark import ParameterSet

ORDERS = (1, 2, 3)


def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


# --------------------------------------------------------------------------------------------
# 1. closed form -> brute force


def test_brute_force_matches_closed_form_dummy_game() -> None:
    game = DummyGame(6, interaction=(1, 4))
    shapley = BruteForceComputer(game).exact_values("SV", 1)
    for player in range(6):
        expected = 1 / 6 + (0.5 if player in (1, 4) else 0.0)
        assert shapley[(player,)] == pytest.approx(expected)


@pytest.mark.parametrize("interaction", [(2,), (0, 3), (1, 2, 4)])
def test_brute_force_matches_closed_form_unanimity_game(interaction: tuple[int, ...]) -> None:
    binary = np.zeros(5, dtype=int)
    binary[list(interaction)] = 1
    computer = BruteForceComputer(UnanimityGame(binary))
    size = len(interaction)
    shapley = computer.exact_values("SV", 1)
    banzhaf = computer.exact_values("BV", 1)
    moebius = computer.exact_values("Moebius", 3)
    for player in range(5):
        member = player in interaction
        assert shapley[(player,)] == pytest.approx(1 / size if member else 0.0)
        assert banzhaf[(player,)] == pytest.approx(1 / 2 ** (size - 1) if member else 0.0)
    for subset_size in ORDERS:
        for subset in combinations(range(5), subset_size):
            assert moebius[subset] == pytest.approx(1.0 if subset == interaction else 0.0)


# --------------------------------------------------------------------------------------------
# 2. brute force -> structured computers


def _assert_agrees_with_brute_force(computer: Computer, *, atol: float = 1e-10) -> int:
    """Compare every supported (index, order) with brute force; return the number compared."""
    brute_force = BruteForceComputer(computer.game)
    compared = 0
    for index in computer.supported_indices():
        for order in ORDERS:
            if not computer.supports(index, order) or not brute_force.supports(index, order):
                continue
            expected = brute_force.exact_values(index, order)
            actual = computer.exact_values(index, order)
            # structured computers may leave out zeros, so compare interaction by interaction
            interactions = sorted({*expected.dict_values, *actual.dict_values})
            assert all(1 <= len(interaction) <= order for interaction in interactions)
            # brute-force FSII solves a weighted least-squares problem (numerical error ~1e-10)
            tolerance = max(atol, 1e-8) if index == "FSII" else atol
            np.testing.assert_allclose(
                [actual[interaction] for interaction in interactions],
                [expected[interaction] for interaction in interactions],
                atol=tolerance,
                err_msg=f"{index} order {order}",
            )
            assert actual.baseline_value == pytest.approx(expected.baseline_value)
            compared += 1
    assert compared > 0
    return compared


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_moebius_computer(seed: int, normalize: bool) -> None:  # noqa: FBT001
    game = SOUM(7, n_basis_games=20, max_interaction_size=4, random_state=seed, normalize=normalize)
    assert (
        _assert_agrees_with_brute_force(MoebiusComputer(game)) == 3 * 7 + 2
    )  # 7 indices (6 of the converter and Moebius itself) x 3 orders + SV, BV


def test_moebius_computer_dummy_and_unanimity_games() -> None:
    _assert_agrees_with_brute_force(MoebiusComputer(DummyGame(5, interaction=(0, 2, 3))))
    _assert_agrees_with_brute_force(MoebiusComputer(DummyGame(5)))  # without an interaction
    _assert_agrees_with_brute_force(MoebiusComputer(UnanimityGame(np.array([1, 1, 0, 0, 1]))))


@pytest.fixture(scope="module")
def tabular() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    return {
        "x": x,
        "y_reg": x[:, 0] * x[:, 1] + x[:, 2] - x[:, 3] ** 2,
        "y_clf": (x[:, 0] + x[:, 1] * x[:, 2] > 0).astype(int),
        "y_multi": np.digitize(x[:, 0] + x[:, 1], [-0.5, 0.5]),
    }


def _requires(package: str | None) -> tuple[pytest.MarkDecorator, ...]:
    if package is None:
        return ()
    return (pytest.mark.skipif(not _installed(package), reason=f"{package} is not installed"),)


def _cases(models: dict[str, tuple[str | None, Callable[[dict], object]]]) -> list[ParameterSet]:
    """One test case per model; models of uninstalled packages are reported as skipped."""
    return [pytest.param(name, marks=_requires(package)) for name, (package, _) in models.items()]


# name -> (the package it needs, a function fitting it on the tabular data)
_TREE_MODELS: dict[str, tuple[str | None, Callable[[dict], object]]] = {
    "dt_reg": (
        None,
        lambda t: DecisionTreeRegressor(max_depth=4, random_state=0).fit(t["x"], t["y_reg"]),
    ),
    "dt_clf": (
        None,
        lambda t: DecisionTreeClassifier(max_depth=4, random_state=0).fit(t["x"], t["y_clf"]),
    ),
    "rf_reg": (
        None,
        lambda t: RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0).fit(
            t["x"], t["y_reg"]
        ),
    ),
    "rf_multi": (
        None,
        lambda t: RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0).fit(
            t["x"], t["y_multi"]
        ),
    ),
    "gb_clf": (
        None,
        lambda t: GradientBoostingClassifier(n_estimators=4, max_depth=2, random_state=0).fit(
            t["x"], t["y_clf"]
        ),
    ),
    "hgb_multi": (
        None,
        lambda t: HistGradientBoostingClassifier(max_iter=4, max_depth=3, random_state=0).fit(
            t["x"], t["y_multi"]
        ),
    ),
    "cat_clf": (
        "catboost",
        lambda t: (
            importlib.import_module("catboost")
            .CatBoostClassifier(iterations=4, depth=3, verbose=0, random_seed=0, thread_count=1)
            .fit(t["x"], t["y_clf"])
        ),
    ),
    "xgb_clf": (
        "xgboost",
        lambda t: (
            importlib.import_module("xgboost")
            .XGBClassifier(n_estimators=4, max_depth=3, random_state=0, n_jobs=1)
            .fit(t["x"], t["y_clf"])
        ),
    ),
    "lgbm_reg": (
        "lightgbm",
        lambda t: (
            importlib.import_module("lightgbm")
            .LGBMRegressor(n_estimators=4, max_depth=3, random_state=0, n_jobs=1, verbose=-1)
            .fit(t["x"], t["y_reg"])
        ),
    ),
}


@pytest.mark.parametrize("name", _cases(_TREE_MODELS))
def test_path_dependent_tree_computer(tabular: dict[str, np.ndarray], name: str) -> None:
    model = _TREE_MODELS[name][1](tabular)
    for normalize in (False, True):
        game = PathDependentTreeGame(model, tabular["x"][7], normalize=normalize)
        atol = 1e-6 if "xgb" in name else 1e-10  # XGBoost thresholds are float32
        _assert_agrees_with_brute_force(PathDependentTreeComputer(game), atol=atol)


@pytest.mark.parametrize("name", _cases(_TREE_MODELS))
def test_interventional_tree_computer(tabular: dict[str, np.ndarray], name: str) -> None:
    model = _TREE_MODELS[name][1](tabular)
    game = InterventionalTreeGame(model, tabular["x"][:15], tabular["x"][20])
    atol = 1e-6 if "xgb" in name else 1e-10
    _assert_agrees_with_brute_force(InterventionalTreeComputer(game), atol=atol)


_MULTICLASS_BOOSTERS: dict[str, tuple[str | None, Callable[[dict], object]]] = {
    "gb": (
        None,
        lambda t: GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=0).fit(
            t["x"], t["y_multi"]
        ),
    ),
    "xgb": (
        "xgboost",
        lambda t: (
            importlib.import_module("xgboost")
            .XGBClassifier(n_estimators=3, max_depth=2, random_state=0, n_jobs=1)
            .fit(t["x"], t["y_multi"])
        ),
    ),
}


@pytest.mark.parametrize("class_index", [0, 1, 2])
@pytest.mark.parametrize("name", _cases(_MULTICLASS_BOOSTERS))
def test_tree_computers_for_every_class_of_multiclass_boosters(
    tabular: dict[str, np.ndarray], name: str, class_index: int
) -> None:
    x = tabular["x"]
    model = _MULTICLASS_BOOSTERS[name][1](tabular)
    path_dependent = PathDependentTreeGame(model, x[3], class_index=class_index)
    _assert_agrees_with_brute_force(PathDependentTreeComputer(path_dependent), atol=1e-6)
    interventional = InterventionalTreeGame(model, x[:10], x[3], class_index=class_index)
    _assert_agrees_with_brute_force(InterventionalTreeComputer(interventional), atol=1e-6)


_BINARY_BOOSTERS: dict[str, tuple[str | None, Callable[[dict], object]]] = {
    "gb": (
        None,
        lambda t: GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=0).fit(
            t["x"], t["y_clf"]
        ),
    ),
    "hgb": (
        None,
        lambda t: HistGradientBoostingClassifier(max_iter=3, max_depth=2, random_state=0).fit(
            t["x"], t["y_clf"]
        ),
    ),
    "xgb": (
        "xgboost",
        lambda t: (
            importlib.import_module("xgboost")
            .XGBClassifier(n_estimators=3, max_depth=2, random_state=0, n_jobs=1)
            .fit(t["x"], t["y_clf"])
        ),
    ),
    "lgbm": (
        "lightgbm",
        lambda t: (
            importlib.import_module("lightgbm")
            .LGBMClassifier(n_estimators=3, max_depth=2, random_state=0, verbose=-1)
            .fit(t["x"], t["y_clf"])
        ),
    ),
    "cat": (
        "catboost",
        lambda t: (
            importlib.import_module("catboost")
            .CatBoostClassifier(iterations=3, depth=2, verbose=0, random_seed=0, thread_count=1)
            .fit(t["x"], t["y_clf"])
        ),
    ),
}


@pytest.mark.parametrize("name", _cases(_BINARY_BOOSTERS))
def test_tree_computers_for_both_classes_of_binary_boosters(
    tabular: dict[str, np.ndarray], name: str
) -> None:
    """A binary booster has one margin, the log-odds of class 1; class 0 explains its negative."""
    x = tabular["x"]
    model = _BINARY_BOOSTERS[name][1](tabular)
    atol = 1e-6 if name == "xgb" else 1e-10  # XGBoost thresholds are float32
    for class_index in (0, 1):
        path_dependent = PathDependentTreeGame(model, x[3], class_index=class_index)
        _assert_agrees_with_brute_force(PathDependentTreeComputer(path_dependent), atol=atol)
        interventional = InterventionalTreeGame(model, x[:10], x[3], class_index=class_index)
        _assert_agrees_with_brute_force(InterventionalTreeComputer(interventional), atol=atol)
    coalitions = np.eye(x.shape[1], dtype=bool)
    class_zero = InterventionalTreeGame(model, x[:10], x[3], class_index=0)(coalitions)
    class_one = InterventionalTreeGame(model, x[:10], x[3], class_index=1)(coalitions)
    np.testing.assert_allclose(class_zero, -class_one, err_msg=name)


@pytest.mark.parametrize("class_index", [0, 1, 2])
def test_nearest_neighbor_computers(tabular: dict[str, np.ndarray], class_index: int) -> None:
    x, y = tabular["x"][:12], tabular["y_multi"][:12]
    point = tabular["x"][100]
    knn = KNeighborsClassifier(n_neighbors=3).fit(x, y)
    weighted = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(x, y)
    radius = RadiusNeighborsClassifier(radius=2.5).fit(x, y)
    for game in (
        KNNGame(knn, point, class_index=class_index),
        WeightedKNNGame(weighted, point, class_index=class_index, n_bits=3),
        ThresholdNNGame(radius, point, class_index=class_index),
    ):
        _assert_agrees_with_brute_force(NearestNeighborComputer(game))


def test_product_kernel_computer(tabular: dict[str, np.ndarray]) -> None:
    x = tabular["x"][:60]
    models = [
        SVC(kernel="rbf", gamma=0.3).fit(x, tabular["y_clf"][:60]),
        SVR(kernel="rbf").fit(x, tabular["y_reg"][:60]),
        GaussianProcessRegressor(kernel=RBF(), random_state=0).fit(x, tabular["y_reg"][:60]),
    ]
    for model in models:
        for normalize in (False, True):
            game = ProductKernelGame(model, tabular["x"][150], normalize=normalize)
            _assert_agrees_with_brute_force(ProductKernelComputer(game))
