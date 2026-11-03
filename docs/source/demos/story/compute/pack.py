"""Write the story demo's data, ``../assets/data.js``: a game of three of a house's features.

The story demo leads up to the orders demo in two steps, with a game small enough to see whole: the
house of the orders demo (``../../orders/compute/housing.py``), valued by an XGBoost model, with
only three of its features playing, its latitude, its longitude and its income, and the other five
absent (at the values of the background houses, as everywhere in that demo). Its value of a
coalition is the prediction of the house with only those features known, as the house's full
decomposition (its Möbius transform, exact) gives it: the baseline and the values of the groups of
those features. Of this game it keeps, rounded, in thousands of dollars (relative to the baseline,
the prediction with no feature known):

- the value of every coalition, ``values``, by its players' indices;
- the Shapley values, ``shapley``, and the Möbius transform, ``moebius``;
- FSII of order 1 (the Shapley values), 2 and 3 (the full decomposition), ``orders``, each with its
  R^2 beyond size under the Shapley distribution, as the orders demo scores its orders.

It writes them as a script that sets ``window.SHAPIQ_STORY``.

Usage::

    python pack.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from shapiq import InteractionValues
from shapiq.game_theory.moebius_converter import MoebiusConverter

HERE = Path(__file__).resolve().parent
ORDERS = HERE.parent.parent / "orders" / "compute"
sys.path.insert(0, str(ORDERS))
from faithfulness import all_coalitions, fsii, predictions, r2  # noqa: E402

PLAYERS = ("Latitude", "Longitude", "Income")  # the house's features that play
# how the demo shows a feature's value: as degrees north or west, or as dollars (in $10,000)
SHOWN = {
    "Latitude": lambda x: f"{x:.2f}°N",
    "Longitude": lambda x: f"{-x:.2f}°W",
    "Income": lambda x: f"${x * 10:.1f}k",
}
THOUSANDS = 100  # the model predicts in $100,000: a value in thousands of dollars


def main() -> None:
    """Write ``../assets/data.js`` from the house's full decomposition."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=ORDERS / "results")
    parser.add_argument("--assets", type=Path, default=HERE.parent / "assets")
    args = parser.parse_args()
    house = json.loads((args.results / "house.json").read_text(encoding="utf-8"))
    full = house["orders"][str(house["largest_order"])]["values"]  # the Möbius transform: exact
    playing = [house["features"].index(name) for name in PLAYERS]

    # the game's Möbius transform: the house's, of the groups of the three features only
    n = len(PLAYERS)
    lookup = {(): 0}
    values = [house["baseline"]]
    for players, value in full:
        if set(players) <= set(playing):
            lookup[tuple(sorted(playing.index(p) for p in players))] = len(values)
            values.append(value)
    moebius = InteractionValues(
        values=np.array(values),
        index="Moebius",
        max_order=n,
        min_order=0,
        n_players=n,
        interaction_lookup=lookup,
        baseline_value=house["baseline"],
    )
    converter = MoebiusConverter(moebius)

    def thousands(value: float) -> float:
        return round(value * THOUSANDS, 2)

    # every coalition's value, relative to the empty coalition's
    terms = {group: moebius[group] for group in lookup}
    coalitions = {
        group: sum(value for part, value in terms.items() if part and set(part) <= set(group))
        for size in range(n + 1)
        for group in itertools.combinations(range(n), size)
    }
    proper, weights = all_coalitions(n)
    game = np.array([coalitions[tuple(np.flatnonzero(row))] + house["baseline"] for row in proper])
    orders = []
    for order in range(1, n + 1):
        explanation = fsii(converter, order)
        orders.append(
            {
                "order": order,
                "values": [
                    [list(group), thousands(value)] for group, value in explanation.items() if group
                ],
                "r2": round(
                    r2(game, predictions(explanation, proper), proper.sum(axis=1), weights), 6
                ),
            }
        )
    shapley = converter("SV", 1)
    data = {
        "players": [
            {"name": name, "value": SHOWN[name](house["x"][index])}
            for name, index in zip(PLAYERS, playing, strict=True)
        ],
        "baseline": thousands(house["baseline"]),
        "prediction": thousands(house["prediction"]),
        "values": [[list(group), thousands(value)] for group, value in coalitions.items()],
        "shapley": [thousands(shapley[(player,)]) for player in range(n)],
        "moebius": [[list(group), thousands(value)] for group, value in terms.items() if group],
        "orders": orders,
    }
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (args.assets / "data.js").write_text(
        "// The story demo's game (see demo.js), written by ../compute/pack.py.\n"
        f"window.SHAPIQ_STORY = {text};\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
