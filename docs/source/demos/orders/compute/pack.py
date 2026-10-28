"""Write the orders demo's data, ``../assets/data.js``, from the results of the other scripts.

Reads the explanations of the molecule (molecule.py), of the houses of California housing
(housing.py), and of how SigLIP 2 matches the FIxLIP demo's elephant and its caption (fixlip.py). It keeps
what the demo draws on its sheets, rounded: the orders 1 to 4 and the full decomposition (k-SII of
order n, the number of players, which equals the Möbius transform; for the houses, of order 7, the
trees' depth, which equals it already and is exact), every main effect, and the
300 strongest interactions of each order (of the molecule, with the centers of its groups where
shapiq's SI graph plot places them); for the houses, the strongest rows of their heatmap. And the
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

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
ORDERS = (1, 2, 3, 4)  # the sheets, above the one of order n (the full decomposition)
# the interactions (of two or more players) drawn on a sheet, most of them weak: a haze around
# the strongest
STRONGEST = 300
COMPACTNESS = 1e10  # how hard a group's center is pulled to its atoms: si_graph_plot's default
ROWS = 14  # the rows of a sheet of the heatmap
SHORT = ["Income", "Age", "Rooms", "Bedrooms", "Population", "Occupancy", "Lat", "Long"]
# the house's features as its tree shows them, left to right: the location in the middle, next to
# the income and the occupancy, with which it interacts the most
TREE_NAMES = ["Income", "Age", "Rooms", "Beds", "Pop.", "Occup.", "Lat", "Long"]
TREE_LAYOUT = [1, 3, 0, 6, 7, 5, 2, 4]
# the models, as the demo's badge shows them: its icon, name, details, and how it was explained
MODELS = {
    "graph": ["graph", "Graph neural network", "GCN with 2 layers", "exact with GraphSHAP-IQ"],
    "tree": ["tree", "Tree ensemble", "XGBoost, 300 trees of depth 7", "exact with TreeExplainer"],
    "black box": ["box", "Black box", "SigLIP 2, a vision-language model", "ProxySHAP"],
}


def explanation(result: dict[str, Any], order: int) -> dict[str, Any]:
    """Return the explanation of an order: k-SII of order n is the one of the highest order."""
    return result["orders"][str(min(order, result["largest_order"]))]


def model(kind: str) -> dict[str, str]:
    """Return the model of an example, as the demo's badge shows it."""
    return dict(zip(("icon", "title", "detail", "method"), MODELS[kind], strict=True))


def sheets(
    result: dict[str, Any], strongest: int | None = STRONGEST, full: int | None = None
) -> list[dict[str, Any]]:
    """Return the sheets of an explanation: per order, its R^2, main effects and interactions.

    Of the interactions, a sheet keeps the ``strongest`` (all of them for ``None``). The last sheet
    is the full decomposition, of order ``full`` (n, the number of players, for ``None``).
    """
    n = result["n_players"]
    full = full or n
    out = []
    for order in (*ORDERS, full):
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
                "full": order == full,
                "r2": round(explanation(result, order)["r2"], 6),
                "first": [round(value, 5) for value in first],
                "groups": [[players, round(value, 5)] for players, value in groups[:strongest]],
                "count": len(values),
            }
        )
    return out


def place_groups(sheet: dict[str, Any], coords: list[list[float]]) -> None:
    """Add to every group of three or more atoms of a sheet the center where its atoms' connections meet.

    The centers are where shapiq's SI graph plot (``si_graph_plot``) puts them: a spring layout of
    the explanation graph with the atoms fixed, in which every group's center is pulled to its atoms
    (all but its last, as there) by its value, and pushed off all atoms and other centers. They do
    not fall onto an atom the way a centroid does, the center of an atom and its three neighbors.
    The centers are in the units of the coordinates.
    """
    xy = np.array(coords)
    low, side = xy.min(axis=0), np.ptp(xy, axis=0).max()
    pos = {atom: (point - low) / side for atom, point in enumerate(xy)}  # as si_graph_plot scales
    graph = nx.Graph()
    graph.add_nodes_from(pos)
    for players, value in sheet["groups"]:
        if len(players) > 2:
            for player in players[:-1]:
                graph.add_edge(player, tuple(players), weight=abs(value) * COMPACTNESS)
    layout = nx.spring_layout(graph, weight="weight", seed=42, pos=pos, fixed=list(pos))
    for group in sheet["groups"]:
        if len(group[0]) > 2:
            group.append([round(float(c), 4) for c in layout[tuple(group[0])] * side + low])


def fits(result: dict[str, Any], full: int | None = None) -> list[list[float]]:
    """Return the order, R^2 and number of values of every order with interactions, and of ``full``.

    ``full`` is the order of the full decomposition: n, the number of players, for ``None``.
    """
    full = full or result["n_players"]
    orders = [*range(1, result["largest_order"] + 1)]
    if full > result["largest_order"]:
        orders.append(full)
    return [
        [
            order,
            round(explanation(result, order)["r2"], 6),
            len(explanation(result, order)["values"]),
        ]
        for order in orders
    ]


def heatmap(result: dict[str, Any], full: int) -> dict[str, Any]:
    """Return the heatmap of the houses: their predictions and, per sheet, the strongest rows.

    The houses are sorted by their prediction. A row is a feature or a group of features, with its
    value for every house, as an integer from -99 to 99 of the sheet's scale (the 98th percentile
    of its absolute values). The last sheet is the full decomposition, of order ``full``.
    """
    columns = np.argsort(result["predictions"])
    out = []
    for order in (*ORDERS, full):
        values = explanation(result, order)["values"]
        strongest = sorted(values.items(), key=lambda row: -np.mean(np.abs(row[1])))[:ROWS]
        matrix = np.array([row[1] for row in strongest])[:, columns]
        scale = float(np.quantile(np.abs(matrix), 0.98))
        out.append(
            {
                "order": order,
                "full": order == full,
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
        "fits": fits(result, full),
        "quartiles": [round(q, 3) for q in result["shapley_r2_quartiles"]],
    }


def house_values(x: list[float]) -> list[str]:
    """Return the features of a house as its tree shows them, under their names."""
    income, age, rooms, bedrooms, population, occupancy, latitude, longitude = x
    return [
        f"${income * 10:.1f}k",  # the median income of the block, in $10,000
        f"{age:.0f} yrs",
        f"{rooms:.1f}",  # per home
        f"{bedrooms:.1f}",  # per home
        f"{population:,.0f}",  # people
        f"{occupancy:.1f}",  # per home
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
    molecule_sheets = sheets(molecule)
    for sheet in molecule_sheets:
        place_groups(sheet, molecule["coords"])
    fixlip["n_players"] = fixlip["grid"] ** 2 + len(fixlip["tokens"])

    data = {
        "examples": [
            {
                "id": "molecule",
                "kind": "graph",
                "label": "Molecule",
                "detail": "GCN",
                "model": model("graph"),
                "atoms": molecule["atoms"],
                "bonds": molecule["bonds"],
                "coords": molecule["coords"],
                "probability": round(molecule["probability"], 3),
                "accuracy": round(molecule["test_accuracy"], 3),
                "budget": molecule["budget"],
                "largest": molecule["largest_order"],
                "sheets": molecule_sheets,
                "fits": fits(molecule),
            },
            {
                "id": "house",
                "kind": "tree",
                "label": "One house",
                "detail": "XGBoost",
                "model": model("tree"),
                "features": TREE_NAMES,
                "values": house_values(house["x"]),
                "layout": TREE_LAYOUT,
                "prediction": round(house["prediction"], 4),
                "baseline": round(house["baseline"], 4),
                "accuracy": round(house["test_r2"], 3),
                "largest": house["largest_order"],
                # the trees' depth: k-SII of that order is already the full decomposition, exact
                "sheets": sheets(house, strongest=None, full=house["largest_order"]),
                "fits": fits(house, full=house["largest_order"]),
            },
            {
                "id": "houses",
                "kind": "heatmap",
                "label": "150 houses",
                "detail": "XGBoost",
                "model": model("tree"),
                "features": SHORT,
                "largest": houses["largest_order"],
                **heatmap(houses, full=houses["largest_order"]),
            },
            {
                "id": "fixlip",
                "kind": "image-text",
                "label": "Photo and caption",
                "detail": "SigLIP 2",
                "model": {
                    **model("black box"),
                    "method": f"ProxySHAP from {fixlip['budget']:,} coalitions",
                },
                "similarity": round(fixlip["similarity"], 4),
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
