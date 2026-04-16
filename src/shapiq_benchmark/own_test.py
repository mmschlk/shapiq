"""Run a small interventional benchmark demo."""

from __future__ import annotations

from shapiq_benchmark.interventional_bench import InterventionalBench
from shapiq_benchmark.pathdependent_bench import PathdependentBench
from shapiq_benchmark.local_xai_bench import LocalXAIBench


bench = InterventionalBench(
        data_str="california_housing",
        model_str="random_forest",
    )

values = bench.exact_values(index="SII", order=2)
print(values)

