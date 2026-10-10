"""Small diagnostic comparison of measured live-oracle time and cached time charges.

This is intentionally limited to deterministic baseline-imputation games with at
most twelve players. It does not claim that one timing ratio transfers to other
games, batch sizes, hardware, GPUs, or estimators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from shapiq_benchmark.execution import hardware
from shapiq_benchmark.families import make_family
from shapiq_benchmark.runner import CountedGame, builtin_factory, provenance, table_game

if TYPE_CHECKING:
    from collections.abc import Callable


class TracedGame(CountedGame):
    """Verify that each paired run requested the same coalitions and batch sizes."""

    def __init__(self, oracle: Callable, n: int, budget: int) -> None:
        """Keep the existing strict budget counter and add a compact call trace."""
        super().__init__(oracle, n, budget)
        self.trace = hashlib.sha256()
        self.batch_sizes = []

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Trace and evaluate each request through the production budget guard."""
        matrix = np.atleast_2d(coalitions).astype(bool)
        self.trace.update(str(matrix.shape).encode())
        self.trace.update(matrix.tobytes())
        self.batch_sizes.append(len(matrix))
        return super().__call__(coalitions)


def compare(
    *,
    n_players: int = 11,
    dataset: str = "breast_cancer",
    model: str = "random_forest",
    methods: tuple[str, ...] = ("KernelSHAP", "LeverageSHAP"),
    relative_budgets: tuple[float, ...] = (2, 8, 64),
    repeats: int = 3,
    model_cache: Path | None = None,
) -> dict:
    """Measure paired estimators, excluding model fitting, imports, and scoring."""
    if not 1 <= n_players <= 12 or not 1 <= repeats <= 10:
        message = "This diagnostic requires at most twelve players and ten repeats."
        raise ValueError(message)
    if any(not math.isfinite(ratio) or ratio <= 0 or ratio > 128 for ratio in relative_budgets):
        message = "Diagnostic budget ratios must be finite and in (0, 128]."
        raise ValueError(message)
    recipe = {
        "family": "local_baseline",
        "dataset": dataset,
        "n_players": n_players,
        "model_profile": model,
        "instance_seed": 0,
    }
    game, metadata = make_family(
        "local_baseline",
        dataset=dataset,
        n_players=n_players,
        model_profile=model,
        instance_seed=0,
        model_cache=str(model_cache) if model_cache else None,
    )
    masks = ((np.arange(2**n_players)[:, None] >> np.arange(n_players)) & 1).astype(bool)
    start = time.perf_counter()
    values = np.asarray(game(masks), dtype=float)
    enumeration_seconds = time.perf_counter() - start
    cached = table_game(values, n_players, np.full(len(values), enumeration_seconds / len(values)))
    specification = {"n_players": n_players, "index": "SV", "order": 1}
    rows = []
    for method in methods:
        for ratio in relative_budgets:
            budget = math.ceil(ratio * n_players)
            for repeat in range(repeats):
                paired = {}
                # Alternate order to reduce a systematic first-run warmup advantage.
                for mode in ("live", "cached") if repeat % 2 == 0 else ("cached", "live"):
                    counted = TracedGame(game if mode == "live" else cached, n_players, budget)
                    random.seed(repeat)
                    np.random.seed(repeat)  # noqa: NPY002 -- match isolated production cells
                    started = time.perf_counter()
                    estimator = builtin_factory(method, specification, repeat)
                    estimate = estimator.approximate(budget=budget, game=counted)
                    seconds = time.perf_counter() - started
                    if counted.exceeded:
                        message = "Estimator exceeded the diagnostic query cap."
                        raise ValueError(message)
                    paired[mode] = (counted, seconds, estimate.dict_values)
                live, live_seconds, live_values = paired["live"]
                cache, cache_seconds, cache_values = paired["cached"]
                coordinates = sorted(live_values.keys() | cache_values.keys())
                left = np.array([live_values.get(key, 0) for key in coordinates])
                right = np.array([cache_values.get(key, 0) for key in coordinates])
                np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-10)
                if live.trace.hexdigest() != cache.trace.hexdigest():
                    message = "Paired runs did not request identical coalitions."
                    raise ValueError(message)
                charged = (
                    max(0.0, cache_seconds - cache.cache_lookup_seconds)
                    + cache.estimated_oracle_seconds
                )
                rows.append(
                    {
                        "method": method,
                        "relative_budget": ratio,
                        "budget": budget,
                        "seed": repeat,
                        "queries": live.queries,
                        "batch_sizes": live.batch_sizes,
                        "measured_live_seconds": live_seconds,
                        "measured_cached_seconds": cache_seconds,
                        "estimated_uncached_seconds": charged,
                        "live_to_estimated_ratio": live_seconds / charged if charged else None,
                        "maximum_estimate_difference": float(np.max(np.abs(left - right))),
                        "coalition_trace_sha256": live.trace.hexdigest(),
                    }
                )
    return {
        "timing_profile": "diagnostic",
        "official_timing": False,
        "hardware": hardware(),
        "source": provenance(),
        "recipe": recipe,
        "model_metadata": metadata,
        "enumeration_seconds": enumeration_seconds,
        "enumeration_batch_size": len(values),
        "records": rows,
        "scope": "Estimator construction and calls; excludes imports, game/model setup and scoring.",
        "limitation": "One deterministic game instance on the recorded CPU; not a runtime correction factor for other games or hardware.",
    }


def main() -> None:
    """Pin one available CPU and save measured diagnostics without publishing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--players", type=int, default=11)
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--model", default="random_forest")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with threadpool_limits(limits=1):
        result = compare(
            n_players=args.players,
            dataset=args.dataset,
            model=args.model,
            model_cache=args.model_cache,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
