"""Cross-method performance benchmark for SV approximators.

A self-contained runner that compares every SV approximator registered in
``shapiq.approximator.SV_APPROXIMATORS`` against ``ExactComputer``
ground truth across a grid of games, budgets, and seeds. Outputs a
long-format CSV plus matplotlib figures for each metric.

The list of approximators is sourced **dynamically** from
``SV_APPROXIMATORS`` at run time via :mod:`benchmark._discovery`. To
benchmark a new approximator:

  1. Merge the branch that ships this file into your feature branch.
  2. Make sure your new class is added to
     ``shapiq.approximator.SV_APPROXIMATORS``.
  3. Run ``python -m benchmark.performance`` — your method appears in
     the comparison automatically.

No edits to this file are required when adding a new approximator.

CLI examples::

    # Full default sweep — every registered SV method,
    # SOUM(n in 6/8/10), 4 budgets, 3 seeds
    python -m benchmark.performance --plot

    # Restrict to one method (output goes into oddshap_bench_<ts>/)
    python -m benchmark.performance --methods OddSHAP --plot

    # Quick smoke run for development
    python -m benchmark.performance --n 6 --budgets 0.25,1.0 --seeds 0

    # Interface probe — instantiate every method and report compatibility
    python -m benchmark.performance --check
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from shapiq import ExactComputer
from shapiq_games.synthetic import SOUM

from ._discovery import (
    construct_for_sv,
    discover_sv_approximator_names,
    load_approximator,
    safe_approximate,
)


# -----------------------------------------------------------------------------
# Metrics — operate on the full SV vector unless noted
# -----------------------------------------------------------------------------


def mean_squared_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.mean((estimated - ground_truth) ** 2))


def mean_absolute_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.mean(np.abs(estimated - ground_truth)))


def sum_squared_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.sum((estimated - ground_truth) ** 2))


def sum_absolute_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.sum(np.abs(estimated - ground_truth)))


def precision_at_k(
    estimated: np.ndarray, ground_truth: np.ndarray, k: int,
) -> float:
    """Fraction of the top-k |truth| features also in the top-k estimate.

    Operates on the singleton portion (positions 1..n+1); the empty-coalition
    baseline at index 0 is dropped.
    """
    est, truth = estimated[1:], ground_truth[1:]
    k = min(k, len(truth))
    if k == 0:
        return float("nan")
    top_truth = set(np.argsort(-np.abs(truth))[:k].tolist())
    top_est = set(np.argsort(-np.abs(est))[:k].tolist())
    return len(top_truth & top_est) / k


def kendall_tau(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Kendall's tau rank correlation on singleton attributions."""
    from scipy.stats import kendalltau

    tau, _ = kendalltau(estimated[1:], ground_truth[1:])
    return float(tau) if not math.isnan(tau) else float("nan")


METRIC_FUNCTIONS: dict[str, Any] = {
    "MSE": mean_squared_error,
    "MAE": mean_absolute_error,
    "SSE": sum_squared_error,
    "SAE": sum_absolute_error,
    "Precision@5": lambda est, truth: precision_at_k(est, truth, k=5),
    "Precision@10": lambda est, truth: precision_at_k(est, truth, k=10),
    "KendallTau": kendall_tau,
}

# Metrics that read as "lower is better" — plotted on log-y.
_LOWER_IS_BETTER = frozenset({"MSE", "MAE", "SSE", "SAE"})


# -----------------------------------------------------------------------------
# Game registry
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSpec:
    name: str
    n: int
    factory: Any  # callable(seed: int) -> Game


def default_game_specs(n_values: list[int]) -> list[GameSpec]:
    specs: list[GameSpec] = []
    for n in n_values:
        specs.append(
            GameSpec(
                name=f"SOUM(n={n})",
                n=n,
                factory=lambda seed, _n=n: SOUM(
                    n=_n,
                    n_basis_games=15,
                    max_interaction_size=3,
                    random_state=seed,
                ),
            )
        )
    return specs


# -----------------------------------------------------------------------------
# Single-cell evaluation
# -----------------------------------------------------------------------------


@dataclass
class CellResult:
    method: str
    game: str
    n: int
    budget: int
    seed: int
    metrics: dict[str, float]
    runtime_seconds: float
    status: str  # "ok" | "skipped:<reason>" | "error:<reason>"


def evaluate_cell(
    method_name: str,
    game_spec: GameSpec,
    budget: int,
    seed: int,
    ground_truth: np.ndarray,
) -> CellResult:
    """Run one (method, game, budget, seed) cell and compute all metrics."""
    approx_cls = load_approximator(method_name)
    if approx_cls is None:
        return CellResult(
            method=method_name, game=game_spec.name, n=game_spec.n,
            budget=budget, seed=seed, metrics={}, runtime_seconds=0.0,
            status="skipped:not_registered",
        )

    estimator, construct_exc = construct_for_sv(
        approx_cls, game_spec.n, random_state=seed,
    )
    if estimator is None:
        reason = f"({type(construct_exc).__name__})" if construct_exc else ""
        return CellResult(
            method=method_name, game=game_spec.name, n=game_spec.n,
            budget=budget, seed=seed, metrics={}, runtime_seconds=0.0,
            status=f"skipped:incompatible_constructor{reason}",
        )

    game = game_spec.factory(seed)
    t0 = time.perf_counter()
    iv, refuse_exc = safe_approximate(estimator, budget, game)
    runtime = time.perf_counter() - t0
    if iv is None:
        return CellResult(
            method=method_name, game=game_spec.name, n=game_spec.n,
            budget=budget, seed=seed, metrics={},
            runtime_seconds=runtime,
            status=f"skipped:refused_regime({type(refuse_exc).__name__})",
        )

    if iv.values.shape != ground_truth.shape:
        return CellResult(
            method=method_name, game=game_spec.name, n=game_spec.n,
            budget=budget, seed=seed, metrics={},
            runtime_seconds=runtime,
            status=f"skipped:shape_mismatch({iv.values.shape}vs{ground_truth.shape})",
        )

    metrics = {
        name: fn(iv.values, ground_truth)
        for name, fn in METRIC_FUNCTIONS.items()
    }
    return CellResult(
        method=method_name, game=game_spec.name, n=game_spec.n,
        budget=budget, seed=seed, metrics=metrics,
        runtime_seconds=runtime, status="ok",
    )


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------


def run_sweep(
    method_names: list[str],
    game_specs: list[GameSpec],
    budget_pcts: list[float],
    seeds: list[int],
    verbose: bool = True,
) -> list[CellResult]:
    """Run the full cross product of (method, game, budget_pct, seed)."""
    results: list[CellResult] = []
    truth_cache: dict[tuple[str, int], np.ndarray] = {}

    total = len(method_names) * len(game_specs) * len(budget_pcts) * len(seeds)
    done = 0

    for spec in game_specs:
        for seed in seeds:
            key = (spec.name, seed)
            if key not in truth_cache:
                truth_game = spec.factory(seed)
                truth_cache[key] = ExactComputer(
                    truth_game, n_players=spec.n,
                )(index="SV").values

            ground_truth = truth_cache[key]
            for budget_pct in budget_pcts:
                budget = max(2, int(budget_pct * 2 ** spec.n))
                for method in method_names:
                    cell = evaluate_cell(
                        method, spec, budget, seed, ground_truth,
                    )
                    results.append(cell)
                    done += 1
                    if verbose:
                        tag = (
                            f"MSE={cell.metrics.get('MSE'):.3e}"
                            if cell.status == "ok" else cell.status
                        )
                        print(
                            f"  [{done:>4}/{total}] {method:<24} "
                            f"{spec.name:<12} budget={budget:>5} seed={seed}  {tag}",
                            file=sys.stderr,
                        )
    return results


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


def write_csv(results: list[CellResult], path: Path) -> None:
    """Long-format CSV: one row per (method, game, n, budget, seed, metric, value)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "game", "n", "budget", "seed",
            "metric", "value", "runtime_seconds", "status",
        ])
        for r in results:
            if r.status != "ok":
                writer.writerow([
                    r.method, r.game, r.n, r.budget, r.seed,
                    "_", "", r.runtime_seconds, r.status,
                ])
                continue
            for metric_name, metric_value in r.metrics.items():
                writer.writerow([
                    r.method, r.game, r.n, r.budget, r.seed,
                    metric_name, metric_value,
                    r.runtime_seconds, r.status,
                ])


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def _aggregate_by_method(
    results: list[CellResult], game_name: str, metric: str,
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """Group results by method; return (budgets, means, stds) per method."""
    by_method: dict[str, dict[int, list[float]]] = {}
    for r in results:
        if r.status != "ok" or r.game != game_name:
            continue
        value = r.metrics.get(metric)
        if value is None or math.isnan(value):
            continue
        by_method.setdefault(r.method, {}).setdefault(r.budget, []).append(value)

    aggregated: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for method, budget_to_values in by_method.items():
        budgets = sorted(budget_to_values)
        means, stds = [], []
        for b in budgets:
            vs = budget_to_values[b]
            means.append(float(np.mean(vs)))
            stds.append(float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0)
        aggregated[method] = (budgets, means, stds)
    return aggregated


def _plot_one_metric(
    results: list[CellResult], game_name: str, metric: str, out_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    aggregated = _aggregate_by_method(results, game_name, metric)
    if not aggregated:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(aggregated):
        budgets, means, stds = aggregated[method]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(budgets, means_arr, marker="o", label=method)
        if stds_arr.max() > 0:
            ax.fill_between(
                budgets, means_arr - stds_arr, means_arr + stds_arr,
                alpha=0.15,
            )

    ax.set_xscale("log")
    if metric in _LOWER_IS_BETTER:
        # symlog tolerates the machine-precision zeros that fully-exact methods reach
        ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel("Budget (log scale)")
    ax.set_ylabel(
        f"{metric}{' (log scale)' if metric in _LOWER_IS_BETTER else ''}",
    )
    ax.set_title(f"{metric} vs budget — {game_name}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_runtime(results: list[CellResult], game_name: str, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    by_method: dict[str, dict[int, list[float]]] = {}
    for r in results:
        if r.status != "ok" or r.game != game_name:
            continue
        by_method.setdefault(r.method, {}).setdefault(
            r.budget, [],
        ).append(r.runtime_seconds)

    if not by_method:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(by_method):
        budgets = sorted(by_method[method])
        means = [float(np.mean(by_method[method][b])) for b in budgets]
        ax.plot(budgets, means, marker="o", label=method)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Budget (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title(f"Runtime vs budget — {game_name}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_all_figures(results: list[CellResult], out_dir: Path) -> int:
    """Save one figure per (game, metric) plus a runtime figure per game."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed; skipping plots.", file=sys.stderr)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    games = sorted({r.game for r in results if r.status == "ok"})
    saved = 0
    for game in games:
        safe = game.replace("(", "_").replace(")", "").replace("=", "")
        for metric in METRIC_FUNCTIONS:
            metric_safe = metric.replace("@", "_at_")
            if _plot_one_metric(
                results, game, metric, out_dir / f"{metric_safe}_{safe}.png",
            ):
                saved += 1
        if _plot_runtime(results, game, out_dir / f"runtime_{safe}.png"):
            saved += 1
    return saved


# -----------------------------------------------------------------------------
# Compatibility probe
# -----------------------------------------------------------------------------


def check_compatibility(method_names: list[str], n: int = 6) -> int:
    """Probe every method for the conformance contract. Prints a table."""
    print(
        f"{'Method':<25} {'Registered':<11} {'Constructible':<14} Notes",
    )
    print("-" * 80)
    for name in method_names:
        cls = load_approximator(name)
        if cls is None:
            print(
                f"{name:<25} {'no':<11} {'-':<14} not exported by shapiq.approximator",
            )
            continue
        est, exc = construct_for_sv(cls, n, random_state=0)
        if est is None:
            if exc is not None:
                short = str(exc).splitlines()[0]
                print(
                    f"{name:<25} {'yes':<11} {'no':<14} "
                    f"{type(exc).__name__}: {short}",
                )
            else:
                print(
                    f"{name:<25} {'yes':<11} {'no':<14} no recognized SV-mode signature",
                )
            continue
        print(f"{name:<25} {'yes':<11} {'yes':<14} OK ({type(est).__name__})")
    return 0


# -----------------------------------------------------------------------------
# Output directory naming
# -----------------------------------------------------------------------------


def _slugify(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def derive_run_name(method_names: list[str], explicit: str | None) -> str:
    """Pick a run-folder name. ``explicit`` overrides; otherwise use the
    single method name (lower-cased + ``_bench``) or ``sv_sweep``.
    """
    if explicit:
        return explicit
    if len(method_names) == 1:
        return f"{_slugify(method_names[0])}_bench"
    return "sv_sweep"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_comma_int(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_comma_float(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return discover_sv_approximator_names()
    return [x.strip() for x in value.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-method performance benchmark for SV approximators. "
            "Auto-discovers all approximators in shapiq.approximator.SV_APPROXIMATORS."
        ),
    )
    parser.add_argument(
        "--methods", default="all", type=_parse_methods,
        help="Comma-separated approximator names, or 'all' (default).",
    )
    parser.add_argument(
        "--n", default="6,8,10", type=_parse_comma_int,
        help="Comma-separated player counts for SOUM games (default: 6,8,10).",
    )
    parser.add_argument(
        "--budgets", default="0.05,0.25,0.5,1.0", type=_parse_comma_float,
        help="Comma-separated budget fractions of 2^n (default: 0.05,0.25,0.5,1.0).",
    )
    parser.add_argument(
        "--seeds", default="0,42,1337", type=_parse_comma_int,
        help="Comma-separated seeds (default: 0,42,1337).",
    )
    parser.add_argument(
        "--name", default=None,
        help=(
            "Run-folder name override. Default derived from --methods "
            "('<method>_bench' for a single method, else 'sv_sweep')."
        ),
    )
    parser.add_argument(
        "--output-root", default="benchmark/results", type=Path,
        help="Parent directory for run folders (default: benchmark/results).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save figures (one per metric per game, plus runtime per game).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-cell progress output.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Probe every discoverable method for interface compatibility and exit.",
    )
    args = parser.parse_args(argv)

    method_names = args.methods
    if args.check:
        return check_compatibility(method_names)

    run_name = derive_run_name(method_names, args.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"

    game_specs = default_game_specs(args.n)
    total = (
        len(method_names) * len(game_specs) * len(args.budgets) * len(args.seeds)
    )
    print(
        f"Sweep: {len(method_names)} methods x {len(game_specs)} games "
        f"x {len(args.budgets)} budgets x {len(args.seeds)} seeds = {total} cells",
        file=sys.stderr,
    )
    print(f"Methods: {', '.join(method_names)}", file=sys.stderr)
    print(f"Output:  {run_dir}", file=sys.stderr)

    results = run_sweep(
        method_names=method_names,
        game_specs=game_specs,
        budget_pcts=args.budgets,
        seeds=args.seeds,
        verbose=not args.quiet,
    )
    write_csv(results, csv_path)
    print(f"CSV written: {csv_path}", file=sys.stderr)

    if args.plot:
        plot_dir = run_dir / "plots"
        saved = plot_all_figures(results, plot_dir)
        print(f"Plots written ({saved} figures): {plot_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
