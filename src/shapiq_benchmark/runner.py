"""Run approximators on a benchmark and score them against the ground truth."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .metrics import compare, faithfulness

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from shapiq.approximator.base import Approximator

    from .benchmark import Benchmark

__all__ = ["build_approximator", "run", "save_results"]


def build_approximator(
    approximator: type[Approximator],
    *,
    n_players: int,
    index: str,
    order: int,
    random_state: int,
) -> Approximator | None:
    """Build an approximator for ``index`` up to ``order``, or ``None`` if it cannot compute it.

    The number of players is always passed. ``index``, ``max_order``, and ``random_state`` are
    passed if the constructor accepts them; approximators that do not take ``max_order`` only
    support order 1. Whether the index is supported comes from the approximator's
    ``valid_indices``.
    """
    valid_indices = getattr(approximator, "valid_indices", ())
    if index not in valid_indices or (index in ("SV", "BV") and order != 1):
        return None
    parameters = inspect.signature(approximator.__init__).parameters
    takes_all = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    kwargs: dict[str, Any] = {"n": n_players}
    if takes_all or "max_order" in parameters:
        kwargs["max_order"] = order
    elif order != 1:
        return None
    if takes_all or "index" in parameters:
        kwargs["index"] = index
    if takes_all or "random_state" in parameters:
        kwargs["random_state"] = random_state
    built = approximator(**kwargs)
    # constructors that swallow keyword arguments may silently ignore the order
    if getattr(built, "max_order", order) != order:
        return None
    return built


def run(
    benchmark: Benchmark,
    approximators: Mapping[str, type[Approximator]] | Sequence[type[Approximator]],
    budgets: Iterable[int],
    *,
    index: str,
    order: int,
    seeds: Iterable[int] = (0,),
    k: int = 10,
    with_faithfulness: bool = False,
) -> pd.DataFrame:
    """Run every approximator for every budget and seed and score it against the ground truth.

    Unsupported combinations and failures are recorded with their status instead of being dropped.

    Args:
        benchmark: The benchmark.
        approximators: Approximator classes, optionally keyed by a display name.
        budgets: The numbers of game evaluations.
        index: The interaction index.
        order: The highest interaction order.
        seeds: The random states of the approximators. Defaults to ``(0,)``.
        k: The ``k`` of the ``@k`` metrics. Defaults to ``10``.
        with_faithfulness: Whether to also compute the (more expensive) faithfulness metric.

    Returns:
        One row per approximator, budget, and seed with the columns ``approximator``, ``index``,
        ``order``, ``budget``, ``seed``, ``status`` (``"ok"``, ``"unsupported"``, or
        ``"failed"``), ``error``, ``runtime_s``, the metrics, and descriptions of the benchmark.
    """
    budgets, seeds = list(budgets), list(seeds)  # they are iterated once per approximator
    if not isinstance(approximators, dict):
        approximators = {approximator.__name__: approximator for approximator in approximators}
    ground_truth = benchmark.exact_values(index, order)
    game = benchmark.game
    context = {
        "game": type(game).__name__,
        "fingerprint": benchmark.fingerprint,
        "n_players": game.n_players,
        "computer": benchmark.computer.name,
        "index": index,
        "order": order,
    }
    rows = []
    for name, approximator_class in approximators.items():
        for budget in budgets:
            for seed in seeds:
                row: dict[str, Any] = {
                    **context,
                    "approximator": name,
                    "budget": budget,
                    "seed": seed,
                }
                try:
                    approximator = build_approximator(
                        approximator_class,
                        n_players=game.n_players,
                        index=index,
                        order=order,
                        random_state=seed,
                    )
                    if approximator is None:
                        rows.append({**row, "status": "unsupported"})
                        continue
                    start = time.perf_counter()
                    estimate = approximator.approximate(budget=budget, game=game)
                    runtime = time.perf_counter() - start
                    metrics = compare(ground_truth, estimate, k=k)
                    if with_faithfulness:
                        metrics["faithfulness"] = faithfulness(game, estimate, random_state=seed)
                    rows.append({**row, "status": "ok", "runtime_s": runtime, **metrics})
                except Exception as error:  # noqa: BLE001 - failures are results, not crashes
                    rows.append(
                        {**row, "status": "failed", "error": f"{type(error).__name__}: {error}"}
                    )
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, path: str | Path) -> Path:
    """Save benchmark results as CSV (``.csv``) or JSON records (any other suffix).

    Args:
        results: The results of :func:`run`.
        path: The output path; parent directories are created.

    Returns:
        The output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        results.to_csv(path, index=False)
    else:
        results.to_json(path, orient="records", indent=2)
    return path
