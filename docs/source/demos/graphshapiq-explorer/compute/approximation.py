"""GraphSHAP-IQ's approximation variant against shapiq's approximators, for the figure ``approximation``.

For each of the explorer's molecules and questions, the 2-SII of the graph game (the GNN's log-odds
with masked atoms set to the baseline, 3 layers) is estimated at equal numbers of model calls:

- GraphSHAP-IQ with ``max_interaction_size`` lambda = 1, 2, ... up to the first lambda at which it
  is exact (one less than the largest receptive field, Corollary D.1 of the paper), with shapiq's
  efficiency routine (the default). Each lambda gives one budget, its number of model calls; it is
  deterministic.
- KernelSHAP-IQ, SVARM-IQ, SHAP-IQ and Permutation Sampling (shapiq's ``KernelSHAPIQ``,
  ``SVARMIQ``, ``SHAPIQ``, ``PermutationSamplingSII``, index k-SII, order 2, defaults otherwise),
  each with every budget of GraphSHAP-IQ and the seeds 0, 1 and 2.

The error is the mean squared error over all 2-SII of one and two atoms against GraphSHAP-IQ's
exact values (as the paper, Appendix G.4). The game is evaluated once on all 2^n maskings (the
molecules have at most 20 atoms), and every method reads its values from this table, so all of
them see the same float32 values of the GNN.

The paper's interaction-informed variants (Appendix D.3) are not in shapiq. For the Monte Carlo
methods (SVARM-IQ, SHAP-IQ, Permutation Sampling), the paper's variant equals setting the SII of
the pairs outside I to zero before the aggregation to 2-SII; the script computes it from the same
runs (``informed``), for the record. KernelSHAP-IQ's variant (pairs left out of the regression)
is not computed.

Writes ``parts/approximation.json``::

    {"methods": [...], "seeds": 3, "cases": {"<molecule>/<task>": {
        "n", "full": 2^n, "exact": the budget of exact GraphSHAP-IQ, "lambdas": [...],
        "budgets": [...], "graphshapiq": [MSE per budget, 0 where exact],
        "<method>": [[MSE of seed 0, 1, 2] per budget],
        "informed": {"<method>": [[...], ...]}, "pairs_outside": pairs outside I}}}

Usage::

    python approximation.py            # the molecules with their own questions (about 10 min)
    python approximation.py --all      # every molecule with every question
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from gnn import GNN, coalition_features, data, graph_game, molecule, task
from shapiq import Game
from shapiq.approximator import SHAPIQ, SVARMIQ, KernelSHAPIQ, PermutationSamplingSII
from shapiq.game_theory.aggregation import aggregate_base_interaction
from shapiq.graph import GraphSHAPIQ

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "approximation.json"
SEEDS = (0, 1, 2)
METHODS = {
    "KernelSHAP-IQ": KernelSHAPIQ,
    "SVARM-IQ": SVARMIQ,
    "SHAP-IQ": SHAPIQ,
    "Permutation Sampling": PermutationSamplingSII,
}
MONTE_CARLO = ("SVARM-IQ", "SHAP-IQ", "Permutation Sampling")


class TableGame(Game):
    """The graph game read from its values on all 2^n maskings (by bitmask)."""

    def __init__(self, values: np.ndarray, n: int) -> None:
        """Keep the table of values."""
        self.table = values
        self.powers = 2 ** np.arange(n, dtype=np.int64)
        super().__init__(n_players=n, normalize=False)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the values of the coalitions (a boolean matrix)."""
        return self.table[np.asarray(coalitions, dtype=np.int64) @ self.powers]


def table(m: dict, t: dict) -> np.ndarray:
    """Return the graph game on all 2^n maskings, by bitmask, with the GNN in float32."""
    gnn = GNN(t, m)
    n = gnn.n
    out = np.empty(2**n)
    chunk = 2**14
    for start in range(0, 2**n, chunk):
        masks = np.arange(start, min(start + chunk, 2**n))
        coalitions = (masks[:, None] >> np.arange(n)[None]) & 1
        out[start : start + len(masks)] = gnn(coalition_features(gnn, coalitions))
    return out


def rounded(value: float) -> float:
    """Return a value with four significant digits."""
    return float(f"{value:.4g}")


def case(m: dict, t: dict) -> dict:
    """Return the errors of all methods at all budgets for a molecule and a question."""
    n = len(m["atoms"])
    _, graph = graph_game(m, t)
    game = TableGame(table(m, t), n)
    exact_gsq = GraphSHAPIQ(game, graph)
    exact = exact_gsq("k-SII", 2)
    inside = set(exact_gsq.moebius.interaction_lookup)
    keys = [s for size in (1, 2) for s in itertools.combinations(range(n), size)]
    truth = np.array([exact[s] for s in keys])
    outside = [s for s in keys if s not in inside]

    def mse(values) -> float:
        return float(np.mean((np.array([values[s] for s in keys]) - truth) ** 2))

    lambdas, budgets, errors = [], [], []
    for lam in range(1, graph.max_receptive_field_size + 1):
        g = GraphSHAPIQ(game, graph, max_interaction_size=lam)
        if budgets and g.budget == budgets[-1]:
            continue
        lambdas.append(lam)
        budgets.append(g.budget)
        errors.append(0.0 if g.exact else mse(g("k-SII", 2)))
        if g.exact:
            break
    result = {
        "n": n,
        "full": 2**n,
        "exact": exact_gsq.budget,
        "lambdas": lambdas,
        "budgets": budgets,
        "graphshapiq": [rounded(e) for e in errors],
        "pairs_outside": len(outside),
        "informed": {},
    }
    for name, cls in METHODS.items():
        rows, informed = [], []
        for budget in budgets:
            row, row_informed = [], []
            for seed in SEEDS:
                if name in MONTE_CARLO:
                    sii = cls(n=n, max_order=2, index="SII", random_state=seed).approximate(budget, game)
                    row.append(mse(aggregate_base_interaction(sii, order=2)))
                    for s in outside:
                        sii[s] = 0.0
                    row_informed.append(mse(aggregate_base_interaction(sii, order=2)))
                else:
                    approx = cls(n=n, max_order=2, index="k-SII", random_state=seed)
                    row.append(mse(approx.approximate(budget, game)))
            rows.append([rounded(e) for e in row])
            informed.append([rounded(e) for e in row_informed])
        result[name] = rows
        if name in MONTE_CARLO:
            result["informed"][name] = informed
    return result


def check_aggregation() -> None:
    """Check that SII aggregated here equals shapiq's k-SII of the same run (dopamine)."""
    m = molecule("dopamine")
    t = task(m["task"])
    n = len(m["atoms"])
    game = TableGame(table(m, t), n)
    for name in MONTE_CARLO:
        cls = METHODS[name]
        a = cls(n=n, max_order=2, index="k-SII", random_state=0).approximate(300, game)
        b = aggregate_base_interaction(
            cls(n=n, max_order=2, index="SII", random_state=0).approximate(300, game), order=2
        )
        diff = max(abs(a[s] - b[s]) for s in itertools.combinations(range(n), 2))
        if diff > 1e-9:
            msg = f"{name}: k-SII and aggregated SII differ by {diff}"
            raise ValueError(msg)


def main() -> None:
    """Compute the cases and write the part (keeping cases computed before)."""
    warnings.simplefilter("ignore")
    check_aggregation()
    every = "--all" in sys.argv
    old = json.loads(OUT.read_text(encoding="utf-8"))["cases"] if OUT.exists() else {}
    cases = dict(old)
    for m in data()["molecules"]:
        for t in data()["tasks"]:
            key = f"{m['id']}/{t['id']}"
            if (t["id"] != m["task"] and not every) or (key in old and "--redo" not in sys.argv):
                continue
            start = time.perf_counter()
            cases[key] = c = case(m, t)
            print(
                f"{key}: n={c['n']} exact {c['exact']:,} of {c['full']:,} lambdas {c['lambdas']} "
                f"[{time.perf_counter() - start:.0f} s]",
                flush=True,
            )
            for k, budget in enumerate(c["budgets"]):
                means = " ".join(f"{name} {np.mean(c[name][k]):.2e}" for name in METHODS)
                print(f"  lambda {c['lambdas'][k]:2d} budget {budget:7,}: GraphSHAP-IQ {c['graphshapiq'][k]:.2e} {means}")
            OUT.write_text(
                json.dumps({"methods": list(METHODS), "seeds": len(SEEDS), "cases": cases}, separators=(",", ":")),
                encoding="utf-8",
            )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
