"""Run a small interventional benchmark demo."""

from __future__ import annotations

from shapiq_benchmark.interventional_bench import InterventionalBench
from shapiq_benchmark.pathdependent_bench import PathdependentBench
from shapiq_benchmark.local_xai_bench import LocalXAIBench
from shapiq.approximator import ProxySHAP
from shapiq_benchmark.metrics import get_all_metrics


bench = LocalXAIBench(
        data_str="california_housing",
        model_str="random_forest",
    )

values = bench.exact_values(index="SII", order=2)
print(values)

game = bench.game

n_features = bench.get_dataset().x_explain.shape[0] #TODO find solution for getting n_features
approximator = ProxySHAP(n=n_features, random_state=42)
approx_values = approximator.approximate(game=game, budget=1000)

metrics = get_all_metrics(values, approx_values, game)
print(metrics)