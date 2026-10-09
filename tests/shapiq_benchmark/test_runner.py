"""Tests of the benchmark, its ground-truth cache, and the runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import shapiq
from shapiq_benchmark import Benchmark, BruteForceComputer, run, save_results
from shapiq_benchmark.computers import PathDependentTreeComputer, UnsupportedComputationError
from shapiq_benchmark.runner import build_approximator
from shapiq_benchmark.setups import (
    DataValuationSetup,
    KNNSetup,
    PathDependentTreeSetup,
    WeightedKNNSetup,
    setup_from_dict,
)
from shapiq_games import SOUM, DummyGame

if TYPE_CHECKING:
    from pathlib import Path


def test_benchmark_defaults_to_the_structured_computer() -> None:
    benchmark = Benchmark(SOUM(30, 20, max_interaction_size=3, random_state=0))
    assert benchmark.computer.name == "moebius"
    assert benchmark.exact_values("k-SII", 2).n_players == 30
    assert "moebius" in repr(benchmark)


def test_benchmark_rejects_a_computer_of_another_game() -> None:
    with pytest.raises(ValueError, match="bound to the benchmark's game"):
        Benchmark(DummyGame(3), computer=BruteForceComputer(DummyGame(3)))


def test_ground_truth_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    setup = KNNSetup(dataset="xor", n_train=8)
    benchmark = Benchmark.from_setup(setup)
    assert benchmark.key == setup.key
    first = benchmark.exact_values("SV", 1)
    directory = tmp_path / "ground_truth" / "knn" / setup.key
    assert sorted(file.name for file in directory.iterdir()) == ["knn_SV_1.json", "setup.json"]
    assert setup_from_dict(json.loads((directory / "setup.json").read_text())) == setup

    calls = []
    rebuilt = Benchmark.from_setup(setup)
    original = rebuilt.computer.exact_values
    monkeypatch.setattr(
        rebuilt.computer, "exact_values", lambda *a: calls.append(a) or original(*a)
    )
    second = rebuilt.exact_values("SV", 1)
    assert calls == []  # served from the cache
    assert second.values.tolist() == pytest.approx(first.values.tolist())  # noqa: PD011

    other = KNNSetup(dataset="xor", n_train=8, random_state=1)
    Benchmark.from_setup(other, cache=False).exact_values("SV", 1)
    assert not (tmp_path / "ground_truth" / "knn" / other.key).exists()
    brute_force = Benchmark.from_setup(setup, BruteForceComputer)
    assert brute_force.computer.name == "brute_force"
    assert "setup=knn" in repr(brute_force)

    results = run(rebuilt, [shapiq.KernelSHAP], budgets=[16], index="SV", order=1)
    assert results[["setup", "key"]].drop_duplicates().to_numpy().tolist() == [["knn", setup.key]]


def test_games_without_setup_are_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    benchmark = Benchmark(DummyGame(4))
    benchmark.exact_values("SV", 1)
    assert benchmark.key is None
    assert not (tmp_path / "ground_truth").exists()


def test_benchmark_falls_back_to_brute_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A defaulted computer is completed by brute force; an explicit one is not."""
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    benchmark = Benchmark.from_setup(PathDependentTreeSetup(dataset="xor"))
    assert benchmark.computer.name == "path_dependent_tree"
    assert benchmark.computer_for("FSII", 2).name == "brute_force"
    assert benchmark.exact_values("FSII", 2).index == "FSII"
    assert any((tmp_path / "ground_truth" / "path_dependent_tree").rglob("brute_force_FSII_2.json"))
    explicit = Benchmark(benchmark.game, PathDependentTreeComputer(benchmark.game))
    assert not explicit.supports("FSII", 2)
    with pytest.raises(UnsupportedComputationError):
        explicit.exact_values("FSII", 2)
    # the weighted KNN explainer needs k > 1, so brute force computes k = 1
    knn = Benchmark.from_setup(
        WeightedKNNSetup(dataset="xor", n_train=8, n_bits=3, model_params={"n_neighbors": 1})
    )
    assert knn.computer.name == "brute_force"


def test_from_setup_passes_the_player_cap_to_the_computer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    setup = DataValuationSetup(dataset="xor", n_players=26)
    with pytest.raises(UnsupportedComputationError, match="capped at 25"):
        Benchmark.from_setup(setup, BruteForceComputer)
    assert Benchmark.from_setup(setup, BruteForceComputer, max_players=26).game.n_players == 26


def test_top_order_estimates_are_scored_on_their_order() -> None:
    """SHAP-IQ estimates FSII of the top order only; it is scored on that order alone."""
    benchmark = Benchmark(SOUM(8, 15, max_interaction_size=2, random_state=0))
    results = run(
        benchmark,
        [shapiq.SHAPIQ, shapiq.RegressionFSII],
        budgets=[2**8],
        index="FSII",
        order=2,
        with_faithfulness=True,
    ).set_index("approximator")
    assert results.loc["SHAPIQ", "scored_orders"] == "2"
    assert results.loc["SHAPIQ", "mse"] < 1e-20
    assert np.isnan(results.loc["SHAPIQ", "faithfulness"])
    assert results.loc["RegressionFSII", "scored_orders"] == "1-2"


def test_build_approximator_respects_signatures() -> None:
    kernel_shap = build_approximator(
        shapiq.KernelSHAP, n_players=5, index="SV", order=1, random_state=0
    )
    assert isinstance(kernel_shap, shapiq.KernelSHAP)
    # KernelSHAP ignores max_order: it cannot approximate interactions, and SV is order 1 only
    assert (
        build_approximator(shapiq.KernelSHAP, n_players=5, index="SV", order=2, random_state=0)
        is None
    )
    assert (
        build_approximator(shapiq.KernelSHAP, n_players=5, index="k-SII", order=2, random_state=0)
        is None
    )
    svarmiq = build_approximator(
        shapiq.SVARMIQ, n_players=5, index="k-SII", order=2, random_state=3
    )
    assert svarmiq is not None
    assert svarmiq.max_order == 2
    assert svarmiq.index == "k-SII"


class _FailingApproximator(shapiq.KernelSHAPIQ):
    def approximate(self, budget, game, *args, **kwargs):
        msg = "boom"
        raise RuntimeError(msg)


def test_run_accepts_one_shot_iterables() -> None:
    benchmark = Benchmark(SOUM(6, 8, random_state=0))
    results = run(
        benchmark,
        [shapiq.KernelSHAPIQ, shapiq.SVARMIQ],
        budgets=(budget for budget in [32, 64]),
        index="k-SII",
        order=2,
        seeds=iter([0, 1]),
    )
    assert len(results) == 2 * 2 * 2


def test_run_has_the_same_columns_whatever_the_outcomes() -> None:
    benchmark = Benchmark(SOUM(6, 8, random_state=0))
    unsupported = run(benchmark, [shapiq.KernelSHAP], budgets=[32], index="k-SII", order=2)
    ok = run(benchmark, [shapiq.KernelSHAPIQ], budgets=[32], index="k-SII", order=2)
    assert list(unsupported.columns) == list(ok.columns)
    assert {"error", "mse", "precision_at_k"} <= set(ok.columns)
    assert unsupported["mse"].isna().all()


def test_run_records_every_outcome(tmp_path: Path) -> None:
    benchmark = Benchmark(SOUM(8, 15, max_interaction_size=3, random_state=0))
    results = run(
        benchmark,
        {
            "kernel_shapiq": shapiq.KernelSHAPIQ,
            "kernel_shap": shapiq.KernelSHAP,
            "failing": _FailingApproximator,
        },
        budgets=[50, 256],
        index="k-SII",
        order=2,
        seeds=[0, 1],
        with_faithfulness=True,
    )
    assert len(results) == 3 * 2 * 2
    status = results.groupby("approximator")["status"].unique().to_dict()
    assert list(status["kernel_shapiq"]) == ["ok"]
    assert list(status["kernel_shap"]) == ["unsupported"]
    assert list(status["failing"]) == ["failed"]
    assert results.loc[results.approximator == "failing", "error"].str.contains("boom").all()
    ok = results[results.status == "ok"]
    # with the full budget, KernelSHAPIQ is exact
    full = ok[ok.budget == 256]
    assert full["mse"].max() < 1e-12
    assert full["faithfulness"].notna().all()
    assert set(results["computer"]) == {"moebius"}

    for suffix in (".csv", ".json"):
        path = save_results(results, tmp_path / "out" / f"results{suffix}")
        loaded = pd.read_csv(path) if suffix == ".csv" else pd.read_json(path)
        assert len(loaded) == len(results)
