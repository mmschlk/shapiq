"""The data of the figure ``arrangement``: catechol, resorcinol and hydroquinone for the PAINS GNN.

The three benzenediols have the same eight heavy atoms (six carbons, two oxygens) and differ only in
where the second hydroxy group sits on the ring: next to the first (catechol, benzene-1,2-diol),
one carbon further (resorcinol, -1,3-diol) or opposite (hydroquinone, -1,4-diol). For each, the
script builds the molecule from its SMILES with :func:`gnn.from_smiles`, turns it so that the ring
lies in the same place in all three (the first hydroxy group at the upper left, the second one
clockwise from it), and explains the PAINS GNN on it with GraphSHAP-IQ: the probability and the
log-odds of "PAINS", the prediction with no atoms, and the complete 2-SII (every atom and every
pair of atoms; 36 terms each).

It needs RDKit (``from_smiles``), and nothing random. Run it from this folder::

    python arrangement.py   # writes parts/arrangement.json

The data of the docs was computed in a virtual environment with RDKit 2026.09, numpy 2.5 and shapiq
from the repository.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import gnn

HERE = Path(__file__).resolve().parent
TASK = "PAINS"
MOLECULES = [  # (id, name, IUPAC name, SMILES): the first oxygen is atom 0 in RDKit's order
    ("catechol", "Catechol", "benzene-1,2-diol", "Oc1ccccc1O"),
    ("resorcinol", "Resorcinol", "benzene-1,3-diol", "Oc1cccc(O)c1"),
    ("hydroquinone", "Hydroquinone", "benzene-1,4-diol", "Oc1ccc(O)cc1"),
]
FIRST = 150.0  # where the first hydroxy group points, in degrees (counterclockwise from the right)


def orient(m: dict) -> dict:
    """Turn (and if needed mirror) a benzenediol so that its ring lies as in the other two.

    The ring's centre goes to (0, 0); the carbon of the first oxygen (atom 0) to the angle FIRST, and
    the ring runs clockwise from it to the carbon of the second oxygen (as seen on the page, whose y
    axis points down). The bonds keep their length of 1.
    """
    xy = np.array([a[1:3] for a in m["atoms"]], dtype=float) * [1, -1]  # y up
    ring = m["rings"][0]
    xy -= xy[ring].mean(axis=0)
    oxygens = [k for k, a in enumerate(m["atoms"]) if a[0] == "O"]
    carbon = [next(j for i, j in ((b[0], b[1]), (b[1], b[0])) if i == o) for o in oxygens for b in m["bonds"] if o in b[:2]]
    turn = np.deg2rad(FIRST) - np.arctan2(*xy[carbon[0]][::-1])
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    xy = xy @ rot.T
    # clockwise from the first carbon to the second: the angle falls
    a1, a2 = (np.arctan2(*xy[c][::-1]) for c in carbon)
    if (a2 - a1) % (2 * np.pi) < np.pi:  # counterclockwise: mirror about the first carbon's axis
        axis = np.array([np.cos(np.deg2rad(FIRST)), np.sin(np.deg2rad(FIRST))])
        xy = 2 * (xy @ axis)[:, None] * axis[None] - xy
    xy *= [1, -1]  # back to the page's y axis
    return {**m, "atoms": [[a[0], *np.round(xy[k], 3).tolist(), *a[3:]] for k, a in enumerate(m["atoms"])]}


def main() -> None:
    """Explain the three benzenediols for PAINS and write ``parts/arrangement.json``."""
    t = gnn.task(TASK)
    out = []
    for mid, name, iupac, smiles in MOLECULES:
        m = orient(gnn.from_smiles(smiles, name=name))
        m["id"] = mid
        g = gnn.explain(m, t)
        moebius = g.moebius.dict_values
        empty = float(moebius.get((), 0.0))
        log_odds = gnn.GNN(t, m).predict()
        k2 = g("k-SII", 2).dict_values
        terms = [[list(S), round(float(v), 4)] for S, v in sorted(k2.items(), key=lambda kv: (len(kv[0]), kv[0])) if S]
        oxygens = tuple(k for k, a in enumerate(m["atoms"]) if a[0] == "O")
        total = sum(v for S, v in k2.items() if S)
        assert abs(total - (log_odds - empty)) < 1e-4, (mid, total, log_odds - empty)
        out.append(
            {
                "id": mid,
                "name": name,
                "iupac": iupac,
                "smiles": smiles,
                "atoms": m["atoms"],
                "bonds": m["bonds"],
                "rings": m["rings"],
                "formula": m["formula"],
                "logOdds": round(log_odds, 4),
                "p": round(float(1 / (1 + np.exp(-log_odds))), 4),
                "empty": round(empty, 4),
                "budget": int(g.budget),
                "oxygens": list(oxygens),
                "kSII2": terms,
            }
        )
        print(
            f"{name}: p = {out[-1]['p']:.4f}, log-odds {log_odds:+.4f}, no atoms {empty:+.4f}, "
            f"O-O {k2[oxygens]:+.4f}, sum of the 2-SII {total:+.4f}, budget {g.budget}"
        )
    target = HERE / "parts" / "arrangement.json"
    target.write_text(json.dumps({"task": TASK, "molecules": out}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
