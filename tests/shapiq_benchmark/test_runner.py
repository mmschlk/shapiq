"""Tests of the benchmark, its ground-truth cache, and the runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import shapiq
from shapiq_benchmark import Benchmark, BruteForceComputer, run, save_results
from shapiq_benchmark.runner import build_approximator
from shapiq_benchmark.setups import KNNSetup, setup_from_dict
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
