"""Compute the data of the illustration "lab-waterfall" (chapter 2.1 of the user guide).

One house of the California housing data (the first, as on the landing page), the gradient boosting
model on six of its features of the illustration "units-and-forms", and four explanations of the
house's prediction that all add up to the same total, the prediction minus the average prediction:

- the explanation game of the house with marginal masking: its prediction for every subset of the
  features, with the other features drawn from 256 background houses, for all 2^6 = 64 subsets;
- "features": the Shapley values of the six features (shapiq's ``ExactComputer``, ``SV``);
- "groups": the Shapley values of the parts of a partition, the house's location (latitude and
  longitude) as one part and the other features alone, in the game whose players are the parts;
- "pairs": the Shapley interactions of order 2 (``k-SII``): a value for each feature and each pair;
- "all": the Möbius transform (``Moebius``): a value for each of the 63 non-empty subsets.

Each explanation is efficient: its values add up to the prediction minus the average. Values are in
$1,000s (the data's target is in $100,000s), rounded to 0.01, and for each explanation, the values
shown in whole $1,000s are rounded so that they add up to the total shown, the prediction shown minus
the average shown (largest remainders).

Run it from the repository's root with the docs' dependencies installed; it writes
``../assets/data.js``::

    python docs/source/user_guide/illustrations/lab-waterfall/compute/compute.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import shapiq

FEATURES = [  # name in the data, label and short label in the illustration, as in units-and-forms
    ("MedInc", "Median income", "Income"),
    ("HouseAge", "House age", "Age"),
    ("AveRooms", "Rooms", "Rooms"),
    ("Population", "Population", "Pop."),
    ("Latitude", "Latitude", "Lat."),
    ("Longitude", "Longitude", "Lon."),
]
# the parts of "groups": the house's location as one part, the other features alone
PARTITION = [("MedInc",), ("HouseAge",), ("AveRooms",), ("Population",), ("Latitude", "Longitude")]
HOUSE = 0  # the explained house: the first of the data
BACKGROUND = 256  # background houses for marginal masking
SEED = 0
THOUSANDS = 100  # the target is in $100,000s; the illustration shows $1,000s
OUT = Path(__file__).resolve().parents[1] / "assets" / "data.js"


def main() -> None:
    """Train the model, explain the house four ways, and write the data."""
    data = fetch_california_housing()
    columns = [list(data.feature_names).index(name) for name, *_ in FEATURES]
    x_all, y_all = data.data[:, columns], data.target
    n = len(FEATURES)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=SEED
    )
    model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED).fit(x_train, y_train)
    r2 = r2_score(y_test, model.predict(x_test))
    house = x_all[HOUSE]

    rng = np.random.default_rng(SEED)
    background = x_train[rng.choice(len(x_train), BACKGROUND, replace=False)]

    def game(coalitions: np.ndarray) -> np.ndarray:
        """Predict the house with the features of each coalition, the others from the background."""
        values = np.empty(len(coalitions))
        for row, coalition in enumerate(coalitions.astype(bool)):
            masked = background.copy()
            masked[:, coalition] = house[coalition]
            values[row] = model.predict(masked).mean()
        return values * THOUSANDS

    names = [name for name, *_ in FEATURES]
    parts = [[names.index(name) for name in part] for part in PARTITION]

    def group_game(coalitions: np.ndarray) -> np.ndarray:
        """The game of the parts: a coalition of parts plays with all the features of its parts."""
        features = np.zeros((len(coalitions), n), dtype=bool)
        for p, part in enumerate(parts):
            features[:, part] |= coalitions[:, [p]].astype(bool)
        return game(features)

    computer = shapiq.ExactComputer(game=game, n_players=n)
    baseline = float(computer.baseline_value)
    prediction = float(game(np.ones((1, n), dtype=bool))[0])
    shapley = computer("SV", order=1)
    ksii = computer("k-SII", order=2)
    moebius = computer("Moebius", order=n)
    group_shapley = shapiq.ExactComputer(game=group_game, n_players=len(parts))("SV", order=1)

    # the units of each explanation, in the order of the illustration: by size, then by features
    sizes = {"pairs": 2, "all": n}
    units = {
        "features": [[i] for i in range(n)],
        "groups": parts,
        **{
            key: [
                list(s)
                for size in range(1, top + 1)
                for s in itertools.combinations(range(n), size)
            ]
            for key, top in sizes.items()
        },
    }
    values = {
        "features": [shapley[(i,)] for i in range(n)],
        "groups": [group_shapley[(p,)] for p in range(len(parts))],
        "pairs": [ksii[tuple(s)] for s in units["pairs"]],
        "all": [moebius[tuple(s)] for s in units["all"]],
    }
    # the total shown is the difference of the prices shown, so that a line of the figure adds up
    total = round(prediction) - round(baseline)
    views = {}
    for key, unit_list in units.items():
        exact = [float(v) for v in values[key]]
        views[key] = {
            "units": unit_list,
            "values": [round(v, 2) for v in exact],
            "shown": _round_to_sum(exact, total),
        }
        print(
            f"{key:9s} {len(unit_list):2d} units, sum {sum(exact):8.3f}, "
            f"shown sum {sum(views[key]['shown'])}"
        )

    out = {
        "source": "California housing, the first house; HistGradientBoostingRegressor on 6 features",
        "r2": round(r2, 2),
        "background": BACKGROUND,
        "baseline": round(baseline, 2),
        "prediction": round(prediction, 2),
        "shown": {"baseline": round(baseline), "prediction": round(prediction), "total": total},
        "features": [
            {"name": name, "label": label, "short": short, "value": round(float(house[i]), 4)}
            for i, (name, label, short) in enumerate(FEATURES)
        ],
        "views": views,
    }
    text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    OUT.write_text(
        "// The data of the illustration lab-waterfall (see illustration.js), written by\n"
        "// ../compute/compute.py. Values are in $1,000s.\n"
        f"window.SHAPIQ_LAB_WATERFALL = {text};\n",
        encoding="utf-8",
    )
    print(
        f"R² {r2:.3f}, average {baseline:.2f}, prediction {prediction:.2f}, "
        f"total {prediction - baseline:.2f} (shown {total}), wrote {OUT}"
    )


def _round_to_sum(values: list[float], total: int) -> list[int]:
    """Round to whole numbers that add up to ``total``: down, then up where the most was lost."""
    shown = [math.floor(v) for v in values]
    rest = total - sum(shown)
    order = sorted(range(len(values)), key=lambda k: values[k] - shown[k], reverse=True)
    for k in order[:rest]:
        shown[k] += 1
    return shown


if __name__ == "__main__":
    main()
