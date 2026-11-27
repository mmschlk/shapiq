"""The data of the figure ``order``: the complete SI-Graph of every explanation, order by order.

The figure draws the SI-Graph of a molecule for a question at the explanation orders k = 1 to 4
(k-SII; k = 1 are the Shapley values) and n (the Möbius interactions), with the strongest terms of
the explorer's data (``data.js`` keeps only those). What it reads beside the drawing needs the
complete explanation, which this script computes with GraphSHAP-IQ for all 21 pairs of a molecule and
a question: for every order, the number of non-zero contributions (sets of one atom or more), their
sum, and the sums of the positive and of the negative ones. By efficiency, the sum is the same for
every order: the log-odds minus the prediction with no atoms. The script checks this (to 1e-5, float32 noise).

Nothing is random. Run it from this folder::

    python order.py   # writes parts/order.json, about 15 seconds
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import gnn

HERE = Path(__file__).resolve().parent
ORDERS = [1, 2, 3, 4, "n"]


def summary(values: dict) -> dict:
    """Return the count, the sum and the positive and negative sums of the non-empty sets' values."""
    terms = [float(v) for S, v in values.items() if S and abs(v) > 1e-12]
    return {
        "count": len(terms),
        "sum": round(sum(terms), 4),
        "positive": round(sum(v for v in terms if v > 0), 4),
        "negative": round(sum(v for v in terms if v < 0), 4),
    }


def main() -> None:
    """Explain every molecule for every question and write ``parts/order.json``."""
    out = {}
    for m in gnn.molecules():
        for t in gnn.data()["tasks"]:
            g = gnn.explain(m, t)
            moebius = g.moebius.dict_values
            empty = float(moebius.get((), 0.0))
            log_odds = gnn.GNN(t, m).predict()
            orders = {}
            for k in ORDERS:
                values = moebius if k == "n" else g("k-SII", k).dict_values
                orders[str(k)] = summary(values)
                total = sum(float(v) for S, v in values.items() if S)
                if abs(total - (log_odds - empty)) > 1e-5:
                    msg = f"{m['id']}/{t['id']} order {k}: sum {total}, log-odds minus empty {log_odds - empty}"
                    raise ValueError(msg)
            out[f"{m['id']}/{t['id']}"] = {
                "logOdds": round(log_odds, 4),
                "empty": round(empty, 4),
                "difference": round(log_odds - empty, 4),
                "n": len(m["atoms"]),
                "orders": orders,
            }
            counts = ", ".join(f"{k}: {orders[str(k)]['count']:,}" for k in ORDERS)
            print(f"{m['id']}/{t['id']}: log-odds {log_odds:+.4f} - empty {empty:+.4f} = {log_odds - empty:+.4f}; terms {counts}")
    target = HERE / "parts" / "order.json"
    target.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
