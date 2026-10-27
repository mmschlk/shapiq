"""Write the orders demo's data, ``../assets/data.js``, from the results of the other scripts.

Reads the molecule's explanation (molecule.py) and the illustrative explanation of the FIxLIP demo's
elephant (fixlip.py), and keeps what the demo draws on its sheets, rounded: the orders 1 to 4 and
the highest (the Möbius transform), every main effect, and the strongest interactions of each order.
It writes them as a script that sets ``window.SHAPIQ_ORDERS`` (a script loads from disk too, where
the browser would not fetch a JSON file).

Usage::

    python pack.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ORDERS = (1, 2, 3, 4)  # the sheets, above the one of the highest order (the Möbius transform)
STRONGEST = 72  # the interactions (of two or more players) drawn on a sheet


def sheets(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the sheets of an explanation: per order, its R^2, main effects and interactions."""
    n = result["n_players"]
    largest = result["largest_order"]
    out = []
    for order in (*ORDERS, largest):
        explanation = result["orders"][str(order)]
        first = [0.0] * n
        groups = []
        for players, value in explanation["values"]:
            if len(players) == 1:
                first[players[0]] = value
            else:
                groups.append([players, value])
        groups.sort(key=lambda group: -abs(group[1]))
        out.append(
            {
                "order": order,
                "moebius": order == largest,
                "r2": round(explanation["r2"], 3),
                "first": [round(value, 5) for value in first],
                "groups": [[players, round(value, 5)] for players, value in groups[:STRONGEST]],
                "count": len(explanation["values"]),
            }
        )
    return out


def main() -> None:
    """Write ``../assets/data.js`` from the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--assets", type=Path, default=HERE.parent / "assets")
    args = parser.parse_args()
    molecule = json.loads((args.results / "molecule.json").read_text(encoding="utf-8"))
    fixlip = json.loads((args.results / "fixlip.json").read_text(encoding="utf-8"))
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
                "sheets": sheets(molecule),
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
                "sheets": sheets(fixlip),
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
