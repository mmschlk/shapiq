"""The data of the talk's figure "faithfulness" (../assets/figures.js), from the orders demo.

The credit applicant of the orders demo (demos/orders/compute/credit.py, results/applicant.json),
explained exactly at every order. For each of the 254 coalitions of its 8 features but the empty
and the full one, the figure shows what the model predicts with only these features present
(the full decomposition, which is the model) against what an explanation of each order predicts
(the sum of its values over the coalition's subsets), with the coalition's probability under the
Shapley distribution. Each order's R^2 is the demo's, which this script checks.

Run from this folder, after ../../../demos/orders/compute/credit.py:

    python faithfulness.py  # rewrites the block FAITHFULNESS in ../assets/figures.js
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEMO = HERE / "../../../demos/orders/compute"
sys.path.insert(0, str(DEMO))

from faithfulness import all_coalitions, predictions, r2  # noqa: E402

data = json.loads((DEMO / "results/applicant.json").read_text())
n = data["n_players"]
coalitions, probabilities = all_coalitions(n)
sizes = coalitions.sum(axis=1)


def explanation(order: str) -> dict[tuple[int, ...], float]:
    """Return the values of the demo's FSII of an order, by group, the empty group's included."""
    values = {tuple(group): value for group, value in data["orders"][order]["values"]}
    values[()] = data["baseline"]
    return values


model = predictions(explanation(str(data["largest_order"])), coalitions)
orders = {}
for order in sorted(data["orders"], key=int):
    y_hat = predictions(explanation(order), coalitions)
    score = r2(model, y_hat, sizes, probabilities)
    if abs(score - data["orders"][order]["r2"]) > 1e-6:  # the demo shows the same R^2
        msg = f"order {order}: R^2 {score}, the demo has {data['orders'][order]['r2']}"
        raise ValueError(msg)
    orders[order] = {"r2": round(score, 4), "y": [round(float(v), 3) for v in y_hat]}

block = {
    "features": data["features"],
    "probability": [round(float(p) / probabilities.max(), 3) for p in probabilities],
    "model": [round(float(v), 3) for v in model],
    "orders": orders,
}
figures = HERE / "../assets/figures.js"
text = figures.read_text()
start, end = "  // FAITHFULNESS (compute/faithfulness.py)\n", "  // /FAITHFULNESS\n"
line = "  const FAITHFULNESS = " + json.dumps(block, separators=(",", ":")) + ";\n"
text = re.sub(
    re.escape(start) + ".*?" + re.escape(end), lambda _: start + line + end, text, flags=re.DOTALL
)
figures.write_text(text)
print({order: value["r2"] for order, value in orders.items()})
