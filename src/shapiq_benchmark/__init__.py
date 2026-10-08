"""Benchmarking for shapiq: a benchmark is a game plus a computer of its exact values.

- :mod:`shapiq_benchmark.computers`: ground-truth computers wrapping the exact algorithms of
  shapiq (brute force, Möbius, path-dependent and interventional trees, nearest neighbors,
  product kernels) and :func:`default_computer` choosing one for a game.
- :class:`Benchmark`: a game and its computer. ``Benchmark(game)`` works for any game;
  :meth:`Benchmark.from_setup` builds the game of a setup and caches its exact values locally.
- :mod:`shapiq_benchmark.setups`: typed recipes that build the games of :mod:`shapiq_games` from
  names (dataset, model, seeds), one per game except the synthetic ones.
- :mod:`shapiq_benchmark.datasets` and :mod:`shapiq_benchmark.models`: the datasets (downloaded on
  first use and cached locally; no data ships with the package) and the seeded model registry
  the setups use.
- :mod:`shapiq_benchmark.metrics`: error, ranking, and faithfulness metrics.
- :func:`run`: runs approximators over budgets and seeds and scores them.

Examples:
    >>> import shapiq
    >>> from shapiq_games import SOUM
    >>> from shapiq_benchmark import Benchmark, run
    >>> benchmark = Benchmark(SOUM(n=12, n_basis_games=30, random_state=0))
    >>> results = run(benchmark, [shapiq.KernelSHAPIQ, shapiq.SVARMIQ], budgets=[200, 1000],
    ...               index="k-SII", order=2, seeds=[0, 1])
"""

from .benchmark import Benchmark
from .computers import (
    BruteForceComputer,
    Computer,
    InterventionalTreeComputer,
    MoebiusComputer,
    NearestNeighborComputer,
    PathDependentTreeComputer,
    ProductKernelComputer,
    UnsupportedComputationError,
    default_computer,
)
from .metrics import compare, faithfulness
from .runner import run, save_results

__all__ = [
    "Benchmark",
    "BruteForceComputer",
    "Computer",
    "InterventionalTreeComputer",
    "MoebiusComputer",
    "NearestNeighborComputer",
    "PathDependentTreeComputer",
    "ProductKernelComputer",
    "UnsupportedComputationError",
    "compare",
    "default_computer",
    "faithfulness",
    "run",
    "save_results",
]
