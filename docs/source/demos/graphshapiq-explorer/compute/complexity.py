"""The model calls of exact GraphSHAP-IQ for one to four layers, for the figure ``complexity``.

For the explorer's seven molecules and the four of the receptive-field figure (benzene, caffeine,
1-nitropyrene, cholesterol, from ``../../graphshapiq-exact/assets/data.js``), the number of
maskings that GraphSHAP-IQ evaluates, |I| = the number of subsets of the receptive fields, for a
GNN with 1, 2, 3 and 4 layers, against all 2^n maskings. It depends on the graph only, not on the
GNN, and is ``GraphSHAPIQ(...).budget`` on a graph with ``n_hops`` = the layers. The script checks
it against the budgets of both demos' data.

Writes ``parts/complexity.json``::

    {"layers": [1, 2, 3, 4], "molecules": [{"id", "name", "n", "explorer": bool,
        "budgets": [|I| for 1..4 layers], "largest": [largest receptive field for 1..4]}, ...]}

Usage::

    python complexity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gnn import data, explanation
from shapiq import Game
from shapiq.graph import GraphModel, GraphSHAPIQ

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "complexity.json"
EXACT = HERE.parents[1] / "graphshapiq-exact" / "assets" / "data.js"
LAYERS = (1, 2, 3, 4)


def exact_molecules() -> list[dict]:
    """Return the molecules of the receptive-field figure (graphshapiq-exact's data.js)."""
    text = EXACT.read_text(encoding="utf-8")
    return json.loads(text[text.index(" = ") + 3 :].rstrip().removesuffix(";"))["molecules"]


def budgets(m: dict) -> tuple[list[int], list[int]]:
    """Return |I| and the largest receptive field for each number of layers."""
    n = len(m["atoms"])
    edges = np.array([[b[0] for b in m["bonds"]], [b[1] for b in m["bonds"]]], dtype=int)
    out, largest = [], []
    for layers in LAYERS:
        graph = GraphModel(edge_index=edges, n_hops=layers, n_nodes=n)
        out.append(GraphSHAPIQ(Game(n_players=n, normalize=False), graph).budget)
        largest.append(graph.max_receptive_field_size)
    return out, largest


def main() -> None:
    """Compute, check and write the part."""
    rows = []
    for m in data()["molecules"]:
        b, largest = budgets(m)
        if b[2] != explanation(m["id"], m["task"])["budget"]:
            msg = f"{m['id']}: 3 layers {b[2]}, data.js {explanation(m['id'], m['task'])['budget']}"
            raise ValueError(msg)
        rows.append({"id": m["id"], "name": m["name"], "n": len(m["atoms"]), "explorer": True, "budgets": b, "largest": largest})
    for m in exact_molecules():
        b, largest = budgets(m)
        if b != [m["budgets"][str(layers)] for layers in LAYERS]:
            msg = f"{m['id']}: {b}, graphshapiq-exact's data.js {m['budgets']}"
            raise ValueError(msg)
        rows.append({"id": m["id"], "name": m["name"], "n": len(m["atoms"]), "explorer": False, "budgets": b, "largest": largest})
    rows.sort(key=lambda row: (row["n"], row["id"]))
    for row in rows:
        print(f"{row['name']}: n={row['n']} 2^n={2 ** row['n']:,} |I| {row['budgets']} largest field {row['largest']}")
    OUT.write_text(json.dumps({"layers": list(LAYERS), "molecules": rows}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
