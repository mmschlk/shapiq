"""The data of the figure ``orders``: how the Möbius interactions of each molecule split by size.

For each of the explorer's seven molecules, with its own question, GraphSHAP-IQ gives the complete
Möbius interactions (MIs) of the graph game (every MI outside the receptive fields is zero). The
script sums the positive and the negative MIs of the sets of 1, 2, 3 and 4 or more atoms, and counts
them. All of them together sum to the log-odds minus the prediction with no atoms (checked to 1e-5).
For comparison it also writes the same sums for the k-SII of order 4 (sizes 1 to 4).

Nothing is random. Run it from this folder::

    python orders.py   # writes parts/orders.json, about 10 seconds
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import gnn

HERE = Path(__file__).resolve().parent
SIZES = [1, 2, 3, 4]  # 4 stands for 4 or more


def by_size(values: dict) -> dict:
    """Return the positive and negative sums and the counts of the non-zero values, by set size."""
    out = {str(s): {"positive": 0.0, "negative": 0.0, "count": 0, "largest": 0.0} for s in SIZES}
    for S, v in values.items():
        if not S or abs(v) <= 1e-12:
            continue
        part = out[str(min(len(S), SIZES[-1]))]
        part["positive" if v > 0 else "negative"] += float(v)
        part["count"] += 1
        part["largest"] = max(part["largest"], abs(float(v)))
    return {s: {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()} for s, p in out.items()}


def main() -> None:
    """Explain the seven molecules for their own questions and write ``parts/orders.json``."""
    out = {}
    for m in gnn.molecules():
        t = gnn.task(m["task"])
        g = gnn.explain(m, t)
        moebius = g.moebius.dict_values
        empty = float(moebius.get((), 0.0))
        log_odds = gnn.GNN(t, m).predict()
        total = sum(float(v) for S, v in moebius.items() if S)
        if abs(total - (log_odds - empty)) > 1e-5:
            msg = f"{m['id']}: sum of the MIs {total}, log-odds minus empty {log_odds - empty}"
            raise ValueError(msg)
        sizes = by_size(moebius)
        out[m["id"]] = {
            "task": t["id"],
            "n": len(m["atoms"]),
            "logOdds": round(log_odds, 4),
            "empty": round(empty, 4),
            "difference": round(log_odds - empty, 4),
            "moebius": sizes,
            "kSII4": by_size(g("k-SII", 4).dict_values),
        }
        print(
            f"{m['id']}/{t['id']}: difference {log_odds - empty:+.4f}; "
            + "; ".join(f"size {s}: {p['positive']:+.2f} {p['negative']:+.2f} ({p['count']})" for s, p in sizes.items())
        )
    target = HERE / "parts" / "orders.json"
    target.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
