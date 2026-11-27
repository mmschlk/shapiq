"""All 2^n Möbius interactions of every molecule, by brute force, for the figure ``sparsity``.

For each of the explorer's seven molecules and each of the three questions, the GNN is evaluated
on all 2^n maskings and the Möbius transform is taken over all of them, once in single precision
(float32, as the demos compute the GNNs) and once in double precision (float64: the same weights,
cast). Every set is marked as inside or outside the set of non-trivial interactions of
Proposition 3.6, I = union over the atoms of the subsets of their receptive fields (3 layers).

The figure shows the absolute values by the size of the set. With up to 2^20 sets (indigo) per
molecule, the part holds them binned: for every size and every bin of ``1 / BINS`` decades of
the absolute value, the number of sets inside and outside I. Exact zeros get their own count.

Writes ``parts/sparsity.json``::

    {"bins": BINS, "floor": FLOOR, "cases": {"<molecule>/<task>": {
        "n", "inside", "outside",                         # numbers of sets
        "f64": {"in": [[size, bin, count], ...], "out": [...], "zero_in", "zero_out",
                "max_in", "max_out", "max_out_size"},     # largest |MI| inside / outside
        "f32": {...}, "check": largest |f64 - GraphSHAP-IQ| inside I,
        "noise": largest |f32 - f64| over all sets, the rounding errors of float32}}}

where ``bin`` is ``floor(log10|v| * BINS)`` and values below ``10**FLOOR`` count in the bin
``FLOOR * BINS``. Run (about two minutes, most of it indigo)::

    python sparsity.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from gnn import GNN, HOPS, coalition_features, data, explain, mask_of, moebius_brute_force
from shapiq.graph import GraphModel

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "sparsity.json"
BINS = 4  # bins per decade
FLOOR = -17  # values below 10**FLOOR (and exact zeros) are counted apart


def values64(gnn: GNN) -> np.ndarray:
    """Return the game on all 2^n maskings, with the GNN computed in float64."""
    steps = [{**s, "w": s["w"].astype(np.float64), "b": s["b"].astype(np.float64)} for s in gnn.steps]
    adjacency = gnn.adjacency.astype(np.float64)
    n = gnn.n
    out = np.empty(2**n)
    chunk = 2**14
    for start in range(0, 2**n, chunk):
        masks = np.arange(start, min(start + chunk, 2**n))
        coalitions = (masks[:, None] >> np.arange(n)[None]) & 1
        h = coalition_features(gnn, coalitions).astype(np.float64)
        for step in steps:
            if step["gather"]:
                h = adjacency @ h
            if step["sum"]:
                h = h.sum(axis=1, keepdims=True)
            h = h @ step["w"].T + step["b"]
            if step["relu"]:
                h = np.maximum(h, 0)
        out[start : start + len(masks)] = h[:, 0, 0]
    return out


def values32(gnn: GNN) -> np.ndarray:
    """Return the game on all 2^n maskings, with the GNN in float32 (as gnn.all_values)."""
    n = gnn.n
    out = np.empty(2**n)
    chunk = 2**14
    for start in range(0, 2**n, chunk):
        masks = np.arange(start, min(start + chunk, 2**n))
        coalitions = (masks[:, None] >> np.arange(n)[None]) & 1
        out[start : start + len(masks)] = gnn(coalition_features(gnn, coalitions))
    return out


def inside_mask(m: dict) -> np.ndarray:
    """Return, for every bitmask, whether the set lies in I (in one receptive field)."""
    n = len(m["atoms"])
    edges = np.array([[b[0] for b in m["bonds"]], [b[1] for b in m["bonds"]]], dtype=int)
    graph = GraphModel(edge_index=edges, n_hops=HOPS, n_nodes=n)
    inside = np.zeros(2**n, dtype=bool)
    for field in graph.receptive_fields:
        inside[mask_of([int(k) for k in field])] = True
    for bit in range(n):  # every subset of a set in I is in I
        view = inside.reshape(-1, 2, 2**bit)
        view[:, 0, :] |= view[:, 1, :]
    return inside


def binned(moebius: np.ndarray, inside: np.ndarray, sizes: np.ndarray) -> dict:
    """Return the counts of |MI| by size and bin, inside and outside I, and the extremes."""
    a = np.abs(moebius)
    zero = a < 10.0**FLOOR
    b = np.full(len(a), FLOOR * BINS, dtype=np.int64)
    b[~zero] = np.floor(np.log10(a[~zero]) * BINS).astype(np.int64)
    result = {}
    for key, where in (("in", inside), ("out", ~inside)):
        pairs = sizes[where] * 1000 + (b[where] - FLOOR * BINS)
        keys, counts = np.unique(pairs, return_counts=True)
        result[key] = [
            [int(k // 1000), int(k % 1000 + FLOOR * BINS), int(c)]
            for k, c in zip(keys, counts, strict=True)
        ]
        result[f"zero_{key}"] = int((zero & where).sum())
    nonempty = sizes > 0
    result["max_in"] = float(a[inside & nonempty].max())
    out = ~inside
    result["max_out"] = float(a[out].max()) if out.any() else None
    result["max_out_size"] = int(sizes[out][np.argmax(a[out])]) if out.any() else None
    return result


def main() -> None:
    """Compute all cases and write the part."""
    warnings.simplefilter("ignore")
    cases = {}
    for m in data()["molecules"]:
        n = len(m["atoms"])
        inside = inside_mask(m)
        sizes = np.array([bin(k).count("1") for k in range(2**n)], dtype=np.int64)
        for t in data()["tasks"]:
            start = time.perf_counter()
            gnn = GNN(t, m)
            m64 = moebius_brute_force(values64(gnn))
            m32 = moebius_brute_force(values32(gnn))
            gsq = explain(m, t).moebius
            check = max(abs(m64[mask_of(s)] - gsq[s]) for s in gsq.interaction_lookup)
            if len(gsq.interaction_lookup) != int(inside.sum()):
                msg = f"{m['id']}: |I| {inside.sum()}, GraphSHAP-IQ {len(gsq.interaction_lookup)}"
                raise ValueError(msg)
            case = {
                "n": n,
                "inside": int(inside.sum()),
                "outside": int((~inside).sum()),
                "f64": binned(m64, inside, sizes),
                "f32": binned(m32, inside, sizes),
                "check": float(f"{check:.2g}"),
                # the rounding of float32: the largest difference of an MI from its float64 value
                "noise": float(f"{np.abs(m32 - m64).max():.3g}"),
            }
            cases[f"{m['id']}/{t['id']}"] = case
            f64, f32 = case["f64"], case["f32"]
            print(
                f"{m['id']}/{t['id']}: n={n} inside {case['inside']:,} outside {case['outside']:,} "
                f"max in {f64['max_in']:.3f} max out f64 {f64['max_out'] or 0:.1e} "
                f"f32 {f32['max_out'] or 0:.1e} (size {f32['max_out_size']}) "
                f"noise {case['noise']:.1e} GraphSHAP-IQ vs f64 {check:.1e} "
                f"[{time.perf_counter() - start:.1f} s]",
                flush=True,
            )
    for case in cases.values():
        for p in ("f64", "f32"):
            for key in ("max_in", "max_out"):
                if case[p][key] is not None:
                    case[p][key] = float(f"{case[p][key]:.4g}")
    OUT.write_text(
        json.dumps({"bins": BINS, "floor": FLOOR, "cases": cases}, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
