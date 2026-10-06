"""Tests of the computer API and of the drift between computers and core shapiq.

The drift tests make sure the indices and orders a computer claims to support are exactly what the
wrapped core algorithm computes: every supported combination must run in core, and every index the
computer does not support must be rejected by core for interactions (order >= 2). Computers reject
``SV`` and ``BV`` beyond order 1 on purpose, although some core algorithms then silently return
other values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

from shapiq.typing import IndexType
from shapiq_benchmark.computers import (
    DEFAULT_MAX_PLAYERS,
    BruteForceComputer,
    Computer,
    InterventionalTreeComputer,
    KNNComputer,
    MoebiusComputer,
    PathDependentTreeComputer,
    ProductKernelComputer,
    UnsupportedComputationError,
    default_computer,
)
from shapiq_games import (
    SOUM,
    DummyGame,
    InterventionalTreeGame,
    KNNGame,
    LocalExplanation,
    PathDependentTreeGame,
    ProductKernelGame,
    RandomTableGame,
    WeightedKNNGame,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ALL_INDICES = get_args(IndexType)


@pytest.fixture(scope="module")
def forest_data() -> tuple[RandomForestRegressor, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 4))
    model = RandomForestRegressor(n_estimators=2, max_depth=3, random_state=0)
    return model.fit(x, x[:, 0] * x[:, 1]), x


def _core_runs(run: Callable[[], object]) -> bool:
    try:
        run()
    except (ValueError, TypeError, NotImplementedError, KeyError):
        return False
    return True


def test_drift_path_dependent_tree(forest_data) -> None:
    from shapiq.tree import TreeExplainer

    model, x = forest_data
    computer = PathDependentTreeComputer(PathDependentTreeGame(model, x[0]))
    for index in ALL_INDICES:
        for order in (1, 2, 3):

            def core(index: str = index, order: int = order) -> None:
                TreeExplainer(model=model, index=index, max_order=order, min_order=1).explain(x[0])

            if computer.supports(index, order):
                assert _core_runs(core), f"supported {index}@{order} fails in core"
            elif order >= 2 and index not in ("SV", "BV"):
                assert not _core_runs(core), (
                    f"core computes {index}@{order}, the computer should support it"
                )


def test_drift_interventional_tree(forest_data) -> None:
    from shapiq.tree.interventional import InterventionalTreeSHAPIQ

    model, x = forest_data
    computer = InterventionalTreeComputer(InterventionalTreeGame(model, x[:10], x[0]))
    for index in ALL_INDICES:
        for order in (1, 2, 3):

            def core(index: str = index, order: int = order) -> None:
                InterventionalTreeSHAPIQ(
                    model=model, data=x[:10], index=index, max_order=order
                ).explain_function(x=x[0])

            if computer.supports(index, order):
                assert _core_runs(core), f"supported {index}@{order} fails in core"
            elif order >= 2 and index not in ("SV", "BV"):
                assert not _core_runs(core), (
                    f"core computes {index}@{order}, the computer should support it"
                )


def test_drift_brute_force_and_moebius() -> None:
    from shapiq import ExactComputer
    from shapiq.game_theory.moebius_converter import MoebiusConverter

    game = SOUM(5, 8, random_state=0)
    brute_force, moebius = BruteForceComputer(game), MoebiusComputer(game)
    exact, converter = (
        ExactComputer(game=game, n_players=5),
        MoebiusConverter(game.moebius_coefficients),
    )
    for index in ALL_INDICES:
        for order in (1, 2, 3):
            if brute_force.supports(index, order):
                assert _core_runs(lambda index=index, order=order: exact(index=index, order=order))
            if moebius.supports(index, order):
                assert _core_runs(lambda index=index, order=order: converter(index, order))
            elif order >= 2 and index not in ("SV", "BV"):
                assert not _core_runs(lambda index=index, order=order: converter(index, order))


def test_supported_indices_come_from_core_declarations() -> None:
    from shapiq import ExactComputer
    from shapiq.explainer.custom_types import ValidNNExplainerIndices
    from shapiq.explainer.product_kernel.product_kernel import ProductKernelSHAPIQIndices
    from shapiq.game_theory.moebius_converter import ValidMoebiusConverterIndices
    from shapiq.tree.quadrature.computer import QuadratureTreeSHAPIndices

    assert BruteForceComputer.supported_indices() == tuple(ExactComputer.valid_indices)
    assert MoebiusComputer.supported_indices() == get_args(ValidMoebiusConverterIndices)
    assert PathDependentTreeComputer.supported_indices() == get_args(QuadratureTreeSHAPIndices)
    assert KNNComputer.supported_indices() == get_args(ValidNNExplainerIndices)
    assert ProductKernelComputer.supported_indices() == get_args(ProductKernelSHAPIQIndices)
    assert "CUSTOM" not in InterventionalTreeComputer.supported_indices()
    assert "CV" not in InterventionalTreeComputer.supported_indices()  # relabelled CHII in core


def test_values_are_order_one_only() -> None:
    computer = BruteForceComputer(DummyGame(4))
    assert computer.supports("SV", 1)
    assert not computer.supports("SV", 2)
    assert not computer.supports("BV", 2)
    assert not computer.supports("k-SII", 0)
    assert not computer.supports("k-SII", 5)
    assert not computer.supports("not-an-index", 1)
    assert computer.supports("ELC", 1)
    assert not computer.supports("ELC", 2)  # the least core is a value, not an interaction
    with pytest.raises(UnsupportedComputationError, match="does not support"):
        computer.exact_values("SV", 2)


def test_output_convention() -> None:
    game = SOUM(5, 10, random_state=1, normalize=True)
    for computer in (BruteForceComputer(game), MoebiusComputer(game)):
        values = computer.exact_values("k-SII", 2)
        assert values.min_order == 1
        assert values.max_order == 2
        assert values.index == "k-SII"
        assert len(values.interaction_lookup) == 5 + 10
        assert () not in values.interaction_lookup
        assert (
            values.baseline_value
            == pytest.approx(game(game.empty_coalition)[0])
            == pytest.approx(0.0)
        )


def test_brute_force_player_cap() -> None:
    with pytest.raises(UnsupportedComputationError, match="capped at 4"):
        BruteForceComputer(RandomTableGame(5), max_players=4)
    assert DEFAULT_MAX_PLAYERS == 20
    big = SOUM(25, 5, random_state=0)
    assert isinstance(default_computer(big), MoebiusComputer)  # structured games have no cap
    with pytest.raises(UnsupportedComputationError):
        BruteForceComputer(big)


def test_computers_reject_foreign_games() -> None:
    with pytest.raises(UnsupportedComputationError, match="cannot compute"):
        MoebiusComputer(RandomTableGame(4))
    with pytest.raises(UnsupportedComputationError, match="cannot compute"):
        PathDependentTreeComputer(DummyGame(3))


def test_default_computer_mapping(forest_data) -> None:
    model, x = forest_data
    knn = KNeighborsClassifier(n_neighbors=2).fit(x[:8], (x[:8, 0] > 0).astype(int))
    weighted = KNeighborsClassifier(n_neighbors=2, weights="distance").fit(
        x[:8], (x[:8, 0] > 0).astype(int)
    )
    from sklearn.svm import SVR

    expected: list[tuple[object, type[Computer]]] = [
        (SOUM(6, 5, random_state=0), MoebiusComputer),
        (DummyGame(4), MoebiusComputer),
        (PathDependentTreeGame(model, x[0]), PathDependentTreeComputer),
        (InterventionalTreeGame(model, x[:5], x[0]), InterventionalTreeComputer),
        (KNNGame(knn, x[9], 1), KNNComputer),
        (WeightedKNNGame(weighted, x[9], 1, n_bits=3), KNNComputer),
        # exact weights are not what the explainer computes: brute force
        (WeightedKNNGame(weighted, x[9], 1), BruteForceComputer),
        (ProductKernelGame(SVR().fit(x, x[:, 0]), x[0]), ProductKernelComputer),
        (LocalExplanation(model, x, x=0), BruteForceComputer),
        (RandomTableGame(4), BruteForceComputer),
    ]
    for game, computer_class in expected:
        assert type(default_computer(game)) is computer_class, type(game).__name__
