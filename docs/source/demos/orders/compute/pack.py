"""Write the orders demo's data, ``../assets/data.js``, from the results of the other scripts.

Reads the explanations of the molecule (molecule.py), of the houses of California housing
(housing.py), and the illustrative explanation of the FIxLIP demo's elephant (fixlip.py). It keeps
what the demo draws on its sheets, rounded: the orders 1 to 4 and the full decomposition (k-SII of
order n, the number of players, which equals the Möbius transform), every main effect, and the
strongest interactions of each order; for the houses, the strongest rows of their heatmap. And the
R^2 and the number of values of every order up to the highest with an interaction, and of order n,
for the plot of the fit by order. It writes them as a script that sets ``window.SHAPIQ_ORDERS`` (a
script loads from disk too, where the browser would not fetch a JSON file).

Usage::

    python pack.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ORDERS = (1, 2, 3, 4)  # the sheets, above the one of order n (the full decomposition)
STRONGEST = 72  # the interactions (of two or more players) drawn on a sheet
ROWS = 14  # the rows of a sheet of the heatmap
SHORT = ["Income", "Age", "Rooms", "Bedrooms", "Population", "Occupancy", "Lat", "Long"]


def explanation(result: dict[str, Any], order: int) -> dict[str, Any]:
    """Return the explanation of an order: k-SII of order n is the one of the highest order."""
    return result["orders"][str(min(order, result["largest_order"]))]


def sheets(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the sheets of an explanation: per order, its R^2, main effects and interactions."""
    n = result["n_players"]
    out = []
    for order in (*ORDERS, n):
        values = explanation(result, order)["values"]
        first = [0.0] * n
        groups = []
        for players, value in values:
            if len(players) == 1:
                first[players[0]] = value
            else:
                groups.append([players, value])
        groups.sort(key=lambda group: -abs(group[1]))
        out.append(
            {
                "order": order,
                "full": order == n,
                "r2": round(explanation(result, order)["r2"], 6),
                "first": [round(value, 5) for value in first],
                "groups": [[players, round(value, 5)] for players, value in groups[:STRONGEST]],
                "count": len(values),
            }
        )
    return out


def fits(result: dict[str, Any]) -> list[list[float]]:
    """Return the order, R^2 and number of values of every order with interactions, and of n."""
    orders = [*range(1, result["largest_order"] + 1)]
    if result["n_players"] > result["largest_order"]:
        orders.append(result["n_players"])
    return [
        [
            order,
            round(explanation(result, order)["r2"], 6),
            len(explanation(result, order)["values"]),
        ]
        for order in orders
    ]


def heatmap(result: dict[str, Any]) -> dict[str, Any]:
    """Return the heatmap of the houses: their predictions and, per sheet, the strongest rows.

    The houses are sorted by their prediction. A row is a feature or a group of features, with its
    value for every house, as an integer from -99 to 99 of the sheet's scale (the 98th percentile
    of its absolute values).
    """
    n = result["n_players"]
    columns = np.argsort(result["predictions"])
    out = []
    for order in (*ORDERS, n):
        values = explanation(result, order)["values"]
        strongest = sorted(values.items(), key=lambda row: -np.mean(np.abs(row[1])))[:ROWS]
        matrix = np.array([row[1] for row in strongest])[:, columns]
        scale = float(np.quantile(np.abs(matrix), 0.98))
        out.append(
            {
                "order": order,
                "full": order == n,
                "r2": round(explanation(result, order)["r2"], 6),
                "count": len(values),
                "scale": round(scale, 4),
                "rows": [
                    [[int(player) for player in key.split(",")], quantized.tolist()]
                    for (key, _), quantized in zip(
                        strongest,
                        np.round(np.clip(matrix / scale, -1, 1) * 99).astype(int),
                        strict=True,
                    )
                ],
            }
        )
    return {
        "predictions": [round(result["predictions"][i], 3) for i in columns],
        "sheets": out,
        "fits": fits(result),
        "quartiles": [round(q, 3) for q in result["shapley_r2_quartiles"]],
    }


def house_values(x: list[float]) -> list[str]:
    """Return the features of a house as the demo shows them."""
    income, age, rooms, bedrooms, population, occupancy, latitude, longitude = x
    return [
        f"${income * 10:.1f}k",  # the median income of the block, in $10,000
        f"{age:.0f} years",
        f"{rooms:.1f} per home",
        f"{bedrooms:.1f} per home",
        f"{population:,.0f} people",
        f"{occupancy:.1f} per home",
        f"{latitude:.2f}°N",
        f"{-longitude:.2f}°W",
    ]


def main() -> None:
    """Write ``../assets/data.js`` from the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--assets", type=Path, default=HERE.parent / "assets")
    args = parser.parse_args()

    def load(name: str) -> dict[str, Any]:
        return json.loads((args.results / f"{name}.json").read_text(encoding="utf-8"))

    molecule, house, houses, fixlip = (
        load("molecule"),
        load("house"),
        load("houses"),
        load("fixlip"),
    )
    molecule["n_players"] = len(molecule["atoms"])
    fixlip["n_players"] = fixlip["grid"] ** 2 + len(fixlip["tokens"])

    data = {
        "examples": [
            {
                "id": "molecule",
                "kind": "graph",
                "label": "Molecule",
                "detail": "GCN",
                "atoms": molecule["atoms"],
                "bonds": molecule["bonds"],
                "coords": molecule["coords"],
                "probability": round(molecule["probability"], 3),
                "accuracy": round(molecule["test_accuracy"], 3),
                "budget": molecule["budget"],
                "largest": molecule["largest_order"],
                "sheets": sheets(molecule),
                "fits": fits(molecule),
            },
            {
                "id": "house",
                "kind": "features",
                "label": "One house",
                "detail": "XGBoost",
                "features": house["features"],
                "values": house_values(house["x"]),
                "prediction": round(house["prediction"], 4),
                "baseline": round(house["baseline"], 4),
                "accuracy": round(house["test_r2"], 3),
                "largest": house["largest_order"],
                "sheets": sheets(house),
                "fits": fits(house),
            },
            {
                "id": "houses",
                "kind": "heatmap",
                "label": "150 houses",
                "detail": "XGBoost",
                "features": SHORT,
                "largest": houses["largest_order"],
                **heatmap(houses),
            },
            {
                "id": "fixlip",
                "kind": "image-text",
                "label": "Photo and caption",
                "detail": "SigLIP 2",
                "illustrative": True,
                "image": "../fixlip/elephant.jpg",  # the FIxLIP demo's photo, next to its assets
                "grid": fixlip["grid"],
                "tokens": fixlip["tokens"],
                "largest": fixlip["largest_order"],
                "sheets": sheets(fixlip),
                "fits": fits(fixlip),
            },
        ]
    }
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (args.assets / "data.js").write_text(
        "// The orders demo's explanations (see demo.js), written by ../compute/pack.py.\n"
        f"window.SHAPIQ_ORDERS = {text};\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
