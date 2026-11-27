"""What Algorithm 1 produces line by line, for the figure ``algorithm``.

For the explorer's seven molecules and a GNN with 1, 2, 3 and 4 layers: |I|, the number of
maskings and MIs of exact GraphSHAP-IQ (lines 1 to 3), from ``GraphSHAPIQ(...).budget`` on a game
that is never evaluated; the largest receptive field and its atom, which the figure draws (the
first atom with the largest field); and, for the SI orders k = 1 to 4, the number of SIs of at most
k atoms (line 4, the same for the model-agnostic computation). All of it depends on the graph only,
not on the GNN. The script checks |I| against an enumeration of the union of the power sets of the
receptive fields and against ``parts/complexity.json``.

Writes ``parts/algorithm.json``::

    {"layers": [1, 2, 3, 4], "orders": [1, 2, 3, 4], "molecules": {"<id>": {"n",
        "calls": [|I| for 1..4 layers], "largest": [...], "focus": [atom for 1..4],
        "field": [[atoms of the largest field] for 1..4], "sis": [SIs of order <= k for k = 1..4]}}}

Usage::

    python algorithm.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
from gnn import data, edge_index
from shapiq import Game
from shapiq.graph import GraphModel, GraphSHAPIQ

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "algorithm.json"
LAYERS = (1, 2, 3, 4)
ORDERS = (1, 2, 3, 4)


class _NeverEvaluated(Game):
    """A game for counting: GraphSHAP-IQ's budget does not evaluate it."""

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        msg = "The budget must not evaluate the game."
        raise AssertionError(msg)


def union_of_power_sets(fields: list[list[int]]) -> int:
    """Return |I|, the number of subsets of the receptive fields, by enumeration."""
    sets = set()
    for field in fields:
        for subset in range(1 << len(field)):
            sets.add(sum(1 << node for b, node in enumerate(field) if subset >> b & 1))
    return len(sets)


def main() -> None:
    """Compute, check and write the part."""
    complexity = json.loads((HERE / "parts" / "complexity.json").read_text(encoding="utf-8"))
    known = {row["id"]: row["budgets"] for row in complexity["molecules"]}
    out = {}
    for m in data()["molecules"]:
        n = len(m["atoms"])
        row = {"n": n, "calls": [], "largest": [], "focus": [], "field": []}
        for layers in LAYERS:
            graph = GraphModel(edge_index=edge_index(m), n_hops=layers, n_nodes=n)
            budget = GraphSHAPIQ(_NeverEvaluated(n_players=n, normalize=False), graph).budget
            fields = [field.tolist() for field in graph.receptive_fields]
            if budget != union_of_power_sets(fields) or budget != known[m["id"]][layers - 1]:
                msg = f"{m['id']}, {layers} layers: |I| = {budget} does not agree"
                raise AssertionError(msg)
            sizes = [len(field) for field in fields]
            focus = int(np.argmax(sizes))
            row["calls"].append(budget)
            row["largest"].append(max(sizes))
            row["focus"].append(focus)
            row["field"].append(fields[focus])
        row["sis"] = [sum(comb(n, s) for s in range(1, k + 1)) for k in ORDERS]
        out[m["id"]] = row
        print(f"{m['id']:15} n = {n:2}  |I| = {row['calls']}  largest = {row['largest']}")
    part = {"layers": list(LAYERS), "orders": list(ORDERS), "molecules": out}
    OUT.write_text(json.dumps(part, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
