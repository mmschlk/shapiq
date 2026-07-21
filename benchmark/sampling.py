"""Benchmark the sampling path: unit generation and end-to-end approximators.

Run with ``uv run python benchmark/sampling.py``. The synthetic game is cheap
on purpose so the numbers isolate sampling overhead; results feed the
before/after table in ``docs/plans/issue-4-sampling-performance.md``.
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from shapiq import (
    SII,
    SV,
    CallableGame,
    PairedSampler,
    PermutationSampling,
    Regression,
)
from shapiq.explainers._permutation import TaylorPlan, WindowPlan
from shapiq.sampling import (
    BanzhafKernelSampler,
    EmptyState,
    PermutationSampler,
    ShapleyKernelSampler,
    UnitScheduleSampler,
)

if TYPE_CHECKING:
    from collections.abc import Callable

N_PLAYERS = 14


def quadratic_game(n_players: int) -> CallableGame:
    """Return a cheap synthetic quadratic game so sampling overhead dominates."""
    key = jax.random.key(0)
    weights = jax.random.normal(key, (n_players,))
    pairs = jax.random.normal(jax.random.fold_in(key, 1), (n_players, n_players))

    def evaluate(coalitions: object) -> jax.Array:
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)  # type: ignore[attr-defined]
        return masks @ weights + 0.5 * jnp.einsum("...i,ij,...j->...", masks, pairs, masks)

    return CallableGame(fn=evaluate, n_players=n_players)


def median_seconds(run: Callable[[], object], repeats: int) -> float:
    """Return the median wall time of a callable in seconds after one warmup."""
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def sampler_only_cases() -> dict[str, UnitScheduleSampler]:
    """Return the sampler-only workloads."""
    return {
        "ShapleyKernelSampler": ShapleyKernelSampler(N_PLAYERS, random_state=0),
        "BanzhafKernelSampler": BanzhafKernelSampler(N_PLAYERS, random_state=0),
        "PairedSampler(ShapleyKernel)": PairedSampler(ShapleyKernelSampler(N_PLAYERS)),
        "PermutationSampler(WindowPlan order=2)": PermutationSampler(
            N_PLAYERS,
            plan=WindowPlan(N_PLAYERS, 2),
        ),
        "PairedSampler(Permutation WindowPlan)": PairedSampler(
            PermutationSampler(N_PLAYERS, plan=WindowPlan(N_PLAYERS, 2)),
        ),
        "PermutationSampler(TaylorPlan order=2)": PermutationSampler(
            N_PLAYERS,
            plan=TaylorPlan(N_PLAYERS, 2),
        ),
    }


def end_to_end_cases(budget: int) -> dict[str, Callable[[], object]]:
    """Return end-to-end workloads sampling a full budget on the synthetic game."""
    game = quadratic_game(N_PLAYERS)

    def regression(*, deduplicate: bool) -> Callable[[], object]:
        def run() -> object:
            approximator = Regression(game, SV(), random_state=0, deduplicate=deduplicate)
            evolved = approximator.sample(budget)
            return jax.block_until_ready(jnp.asarray(evolved.state.values))  # type: ignore[attr-defined]

        return run

    def permutation(*, deduplicate: bool) -> Callable[[], object]:
        def run() -> object:
            approximator = PermutationSampling(
                game,
                SII(order=2),
                random_state=0,
                deduplicate=deduplicate,
            )
            evolved = approximator.sample(budget)
            return jax.block_until_ready(jnp.asarray(evolved.state.values))  # type: ignore[attr-defined]

        return run

    def split_regression(calls: int) -> Callable[[], object]:
        def run() -> object:
            approximator = Regression(game, SV(), random_state=0, deduplicate=True)
            for _ in range(calls):
                approximator = approximator.sample(budget // calls)
            return jax.block_until_ready(jnp.asarray(approximator.state.values))  # type: ignore[attr-defined]

        return run

    return {
        "Regression(SV) paired": regression(deduplicate=False),
        "Regression(SV) paired dedup": regression(deduplicate=True),
        "PermutationSampling(SII order=2)": permutation(deduplicate=False),
        "PermutationSampling(SII order=2) dedup": permutation(deduplicate=True),
        "Regression(SV) dedup, 64 split calls": split_regression(64),
    }


def main() -> None:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=4096, help="coalition evaluations per run")
    parser.add_argument("--repeats", type=int, default=3, help="timed repetitions (median)")
    args = parser.parse_args()
    budget: int = args.budget
    repeats: int = args.repeats

    print(f"jax {jax.__version__} on {jax.default_backend()}, n_players={N_PLAYERS}")
    print(f"budget={budget}, repeats={repeats} (median), one warmup run each")
    print()
    print("| workload | total ms | us/eval |")
    print("|---|---|---|")

    state = EmptyState()
    for name, sampler in sampler_only_cases().items():

        def run(sampler: UnitScheduleSampler = sampler) -> object:
            coalitions, _ = sampler.sample(state, budget)
            return jax.block_until_ready(jnp.asarray(coalitions.to_dense()))

        seconds = median_seconds(run, repeats)
        print(f"| sampler-only: {name} | {seconds * 1e3:.1f} | {seconds / budget * 1e6:.1f} |")

    game = quadratic_game(N_PLAYERS)
    masks = jax.random.bernoulli(jax.random.key(2), 0.5, (budget, N_PLAYERS))
    from shapiq import DenseCoalitionArray  # noqa: PLC0415 - optional-cost import for one row

    game_seconds = median_seconds(
        lambda: jax.block_until_ready(jnp.asarray(game(DenseCoalitionArray(masks)))),
        repeats,
    )
    print(f"| game-only: quadratic on {budget} masks | {game_seconds * 1e3:.1f} | "
          f"{game_seconds / budget * 1e6:.1f} |")

    for name, run in end_to_end_cases(budget).items():
        seconds = median_seconds(run, repeats)
        print(f"| end-to-end: {name} | {seconds * 1e3:.1f} | {seconds / budget * 1e6:.1f} |")


if __name__ == "__main__":
    main()
