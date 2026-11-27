"""The data of the figure ``pipeline``: from a molecule to the GNN's log-odds, step by step.

For every molecule and every question (the figure follows the explorer's question), the GNN is run
on the unmasked molecule and the script writes

- ``focus``: the atom whose receptive field the figure grows over the three layers (``FOCUS``,
  chosen per molecule: a hetero atom of the substructure the molecule is about);
- ``x``: the focus atom's node features (one-hot over the elements, ``gnn.symbols()``);
- ``h``: the focus atom's embedding after each of the three GIN layers (32 values each);
- ``terms``: each atom's term of the log-odds (its last embedding through the linear readout,
  ``gnn.GNN.atom_terms``), and ``bias``, the readout's bias: the log-odds are their sum, since the
  sum pooling and the readout are linear;
- ``logOdds`` and ``p``: the GNN's log-odds and probability of "yes".

Usage (from this folder)::

    python pipeline.py      # writes parts/pipeline.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import gnn

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "pipeline.json"

# the atom whose receptive field grows in the figure (index into the molecule's atoms, from 0)
FOCUS = {
    "dinitrobenzene": 6,  # N of a nitro group
    "dopamine": 7,  # O of the catechol
    "serotonin": 5,  # N of the indole
    "indigo": 4,  # N of an indole-like ring
    "benzimidazole": 4,  # N of the imidazole
    "aminofluorene": 0,  # N of the amino group
    "aspirin": 3,  # O of the ester
}


def r(values: np.ndarray | float, digits: int = 4) -> list | float:
    """Return values rounded for the JSON."""
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def main() -> None:
    """Run every GNN on every molecule and write the figure's part."""
    out: dict[str, dict] = {}
    for m in gnn.molecules():
        k = FOCUS[m["id"]]
        x = gnn.features(m)
        for t in gnn.data()["tasks"]:
            net = gnn.GNN(t, m)
            layers = net.layers(x)
            terms = net.atom_terms(x).astype(float)
            log_odds = float(net(x[None])[0])
            # the readout is linear: the terms and the bias are the log-odds (float32 rounding)
            assert abs(terms.sum() + net.readout_b - log_odds) < 1e-4
            out[f"{m['id']}/{t['id']}"] = {
                "focus": k,
                "x": r(x[k], 0),
                "h": [r(h[k], 3) for h in layers],
                "terms": r(terms),
                "bias": r(net.readout_b),
                "logOdds": r(log_odds),
                "p": r(1 / (1 + np.exp(-log_odds)), 5),
            }
            if m["id"] == "dinitrobenzene" and t["id"] == "mutagenic":
                print(
                    f"dinitrobenzene/mutagenic: log-odds {log_odds:.4f}, bias {net.readout_b:.4f}, "
                    f"terms {np.round(terms, 3).tolist()}"
                )
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(out)} explanations")


if __name__ == "__main__":
    main()
