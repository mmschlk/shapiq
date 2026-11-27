"""The data of the figure ``maskings``: four maskings of the molecule and the GNN's answer for each.

For every molecule and every question (the figure follows the explorer's question), the graph game
is evaluated on four coalitions: all atoms, no atoms, all atoms but a meaningful substructure
(``SUBSTRUCTURE``, chosen per molecule), and a random masking (``SEED``: half of the atoms, rounded
down, are masked, the same for every question). A masked atom takes the question's baseline
features, as in the explorer. Each masking holds the masked atoms and the GNN's log-odds and
probability of "yes".

Usage (from this folder)::

    python maskings.py      # writes parts/maskings.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import gnn

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "maskings.json"
SEED = 3

# the substructure removed in the third masking: its name (the panel's title) and its atoms (from 0)
SUBSTRUCTURE = {
    "dinitrobenzene": ("No nitro groups", [6, 7, 8, 9, 10, 11]),
    "dopamine": ("No catechol OH", [7, 9]),
    "serotonin": ("No indole N", [5]),
    "indigo": ("No N atoms", [4, 13]),
    "benzimidazole": ("One N masked", [6]),
    "aminofluorene": ("No amino group", [0]),
    "aspirin": ("No acetoxy group", [0, 1, 2, 3]),
}


def main() -> None:
    """Evaluate the four maskings of every molecule for every question and write the part."""
    out: dict[str, dict] = {}
    rng = np.random.default_rng(SEED)
    for m in gnn.molecules():
        n = len(m["atoms"])
        name, removed = SUBSTRUCTURE[m["id"]]
        random = sorted(int(k) for k in rng.permutation(n)[: n // 2])
        maskings = [
            ("All atoms", []),
            ("No atoms", list(range(n))),
            (name, removed),
            ("Random masking", random),
        ]
        for t in gnn.data()["tasks"]:
            net = gnn.GNN(t, m)
            rows = []
            for title, masked in maskings:
                v = net.predict([k for k in range(n) if k not in masked])
                rows.append(
                    {
                        "title": title,
                        "masked": masked,
                        "logOdds": round(v, 4),
                        "p": round(float(1 / (1 + np.exp(-v))), 5),
                    }
                )
            out[f"{m['id']}/{t['id']}"] = rows
            if t["id"] == m["task"]:
                print(m["id"], t["id"], [(row["title"], row["logOdds"], row["p"]) for row in rows])
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(out)} explanations")


if __name__ == "__main__":
    main()
