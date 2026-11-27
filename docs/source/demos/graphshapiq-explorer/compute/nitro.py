"""The data of the figure ``nitro``: an interaction of two atoms, measured on the model.

For every molecule and every question (the figure follows the explorer's question), the pair of
atoms with the strongest 2-SII of GraphSHAP-IQ (the first in atom order among equally strong
pairs: for 1,3-dinitrobenzene a nitrogen and an oxygen of one nitro group). For the pair (i, j)
and maskings T of the other atoms, the script measures the changes of the log-odds when i, j or
both are added, and the discrete derivative

    Delta_ij(T) = nu(T + ij) - nu(T + i) - nu(T + j) + nu(T)

``alone``: with all other atoms masked (T empty). Then Delta_ij is the Möbius interaction of the
pair, ``moebius``.

``random``: the average over ``SAMPLES`` random maskings T of the other atoms (seed ``SEED``). The
number of other atoms that keep their features is drawn uniformly from 0 to n - 2, then the atoms.
This is the distribution of T in the Shapley interaction index, so the average discrete derivative
estimates the SII of the pair, which equals its 2-SII (the 2-SII of a pair is the SII).

Usage (from this folder)::

    python nitro.py         # writes parts/nitro.json
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

import gnn

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "nitro.json"
SEED = 0
SAMPLES = 256


def main() -> None:
    """Measure the strongest pair of every molecule for every question and write the part."""
    warnings.simplefilter("ignore")
    out: dict[str, dict] = {}
    tasks = gnn.data()["tasks"]
    for mk, m in enumerate(gnn.molecules()):
        n = len(m["atoms"])
        for tk, t in enumerate(tasks):
            explained = gnn.explain(m, t)
            sii = explained("k-SII", 2)
            pairs = [(i, j) for i, j in combinations(range(n), 2)]
            values = {p: float(sii[p]) for p in pairs}
            top = max(abs(v) for v in values.values())
            # the strongest pair; ties (symmetric atoms) go to the first pair in atom order
            i, j = next(p for p in pairs if abs(values[p]) >= top - 1e-6)
            net = gnn.GNN(t, m)
            others = [k for k in range(n) if k not in (i, j)]
            rng = np.random.default_rng([SEED, mk, tk])
            kept = []
            for _ in range(SAMPLES):
                size = rng.integers(0, n - 1)  # 0 .. n - 2 other atoms keep their features
                kept.append(rng.choice(others, size=size, replace=False))
            # the four coalitions of each masking T: T, T + i, T + j, T + ij
            coalitions = np.zeros((SAMPLES + 1, 4, n), dtype=bool)
            for s, T in enumerate([np.array([], dtype=int), *kept]):
                coalitions[s, :, T] = True
                coalitions[s, 1, i] = coalitions[s, 3, i] = True
                coalitions[s, 2, j] = coalitions[s, 3, j] = True
            v = net(gnn.coalition_features(net, coalitions.reshape(-1, n))).reshape(-1, 4)
            gains = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=1)
            delta = v[:, 3] - v[:, 1] - v[:, 2] + v[:, 0]
            alone, sampled = gains[0], gains[1:]
            row = {
                "pair": [i, j],
                "alone": {
                    "values": np.round(v[0], 4).tolist(),  # nu(empty), nu(i), nu(j), nu(ij)
                    "i": round(float(alone[0]), 4),
                    "j": round(float(alone[1]), 4),
                    "both": round(float(alone[2]), 4),
                    "interaction": round(float(delta[0]), 4),
                },
                "random": {
                    "i": round(float(sampled[:, 0].mean()), 4),
                    "j": round(float(sampled[:, 1].mean()), 4),
                    "both": round(float(sampled[:, 2].mean()), 4),
                    "interaction": round(float(delta[1:].mean()), 4),
                    "se": round(float(delta[1:].std(ddof=1) / np.sqrt(SAMPLES)), 4),
                },
                "moebius": round(float(explained.moebius[(i, j)]), 4),
                "sii": round(values[(i, j)], 4),
                "samples": SAMPLES,
            }
            out[f"{m['id']}/{t['id']}"] = row
            if t["id"] == m["task"]:
                print(m["id"], t["id"], json.dumps(row))
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(out)} explanations")


if __name__ == "__main__":
    main()
