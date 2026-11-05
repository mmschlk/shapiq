"""Compute the data of the illustration "units-and-forms" (chapter 2.1 of the user guide).

One house of the California housing data (the first, as on the landing page), a gradient boosting
model on six of its features, and the explanations that the illustration shows for it:

- the explanation game of the house with marginal masking: its prediction for every subset of the
  features, with the other features drawn from background houses, for all 2^6 = 64 subsets;
- numbers: the Shapley values of the features, the Shapley values of groups of features taken as
  one player, and the pairwise Shapley interactions (k-SII of order 2), with shapiq's
  ``ExactComputer``;
- curves: the house's prediction as one feature varies (its ICE curve), as two features vary (a
  surface), and the part of that surface that neither feature explains alone (the pure interaction).

Run it from the repository's root with the docs' dependencies installed; it writes
``../assets/data.js``::

    python docs/source/user_guide/illustrations/units-and-forms/compute/compute.py
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

FEATURES = [  # name in the data, label in the illustration, unit of the value
    ("MedInc", "Median income", "\u00d7$10k"),
    ("HouseAge", "House age", "years"),
    ("AveRooms", "Rooms", "per household"),
    ("Population", "Population", "of the block"),
    ("Latitude", "Latitude", "°N"),
    ("Longitude", "Longitude", "°E"),
]
HOUSE = 0  # the explained house: the first of the data
BACKGROUND = 256  # background houses for marginal masking
GRID_1D = 40  # points of a curve
GRID_2D = 20  # points per axis of a surface
QUANTILES = (0.02, 0.98)  # the range of a feature's grid
SEED = 0
OUT = Path(__file__).resolve().parents[1] / "assets" / "data.js"


def main() -> None:
    """Train the model, explain the house, and write the data."""
    data = fetch_california_housing()
    columns = [list(data.feature_names).index(name) for name, _, _ in FEATURES]
    x_all, y_all = data.data[:, columns], data.target
    n = len(FEATURES)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=SEED
    )
    model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED).fit(x_train, y_train)
    r2 = r2_score(y_test, model.predict(x_test))
    house = x_all[HOUSE]
    prediction = float(model.predict(house[None])[0])

    rng = np.random.default_rng(SEED)
    background = x_train[rng.choice(len(x_train), BACKGROUND, replace=False)]

    def game(coalitions: np.ndarray) -> np.ndarray:
        """Predict the house with the features of each coalition, the others from the background."""
        values = np.empty(len(coalitions))
        for row, coalition in enumerate(coalitions.astype(bool)):
            masked = background.copy()
            masked[:, coalition] = house[coalition]
            values[row] = model.predict(masked).mean()
        return values

    subsets = [s for size in range(n + 1) for s in itertools.combinations(range(n), size)]
    coalitions = np.array([[i in s for i in range(n)] for s in subsets])
    table = dict(zip(subsets, game(coalitions), strict=True))

    computer = shapiq.ExactComputer(game=game, n_players=n)
    shapley = computer("SV", order=1)
    ksii = computer("k-SII", order=2)

    def group_value(group: tuple[int, ...]) -> float:
        """The Shapley value of a group taken as one player, in the game with the group merged."""
        others = [i for i in range(n) if i not in group]
        m = len(others) + 1
        value = 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for rest in itertools.combinations(others, size):
                with_group = tuple(sorted(rest + group))
                value += weight * (table[with_group] - table[rest])
        return value

    def grid(feature: int, points: int) -> np.ndarray:
        low, high = np.quantile(x_all[:, feature], QUANTILES)
        return np.linspace(low, high, points)

    def predict_with(changes: dict[int, np.ndarray]) -> np.ndarray:
        """Predict the house with the features of ``changes`` set to their (broadcast) values."""
        shape = np.broadcast_shapes(*(v.shape for v in changes.values()))
        rows = np.repeat(house[None], math.prod(shape), axis=0)
        for feature, values in changes.items():
            rows[:, feature] = np.broadcast_to(values, shape).ravel()
        return model.predict(rows).reshape(shape)

    curves = []
    for i in range(n):
        z = grid(i, GRID_1D)
        curves.append({"x": z.tolist(), "y": predict_with({i: z}).tolist()})

    surfaces, interactions = {}, {}
    for i, j in itertools.combinations(range(n), 2):
        zi, zj = grid(i, GRID_2D), grid(j, GRID_2D)
        both = predict_with({i: zi[:, None], j: zj[None, :]})  # rows: feature i, columns: j
        only_i = predict_with({i: zi})[:, None]
        only_j = predict_with({j: zj})[None, :]
        key = f"{i}-{j}"
        surfaces[key] = {"x": zi.tolist(), "y": zj.tolist(), "z": both.tolist()}
        interactions[key] = (both - only_i - only_j + prediction).tolist()

    out = {
        "source": "California housing, the first house; HistGradientBoostingRegressor on 6 features",
        "r2": round(r2, 3),
        "features": [
            {"name": name, "label": label, "unit": unit, "value": float(house[i])}
            for i, (name, label, unit) in enumerate(FEATURES)
        ],
        "prediction": prediction,
        "baseline": float(table[()]),
        "shapley": [float(shapley[(i,)]) for i in range(n)],
        "groups": {"-".join(map(str, s)): group_value(s) for s in subsets if len(s) >= 2},
        "interactions": {
            f"{i}-{j}": float(ksii[(i, j)]) for i, j in itertools.combinations(range(n), 2)
        },
        "curves": curves,
        "surfaces": surfaces,
        "pure": interactions,
    }
    rounded = _round(out)
    rounded["features"] = out["features"]  # the house's values exactly, as in the data
    text = json.dumps(rounded, separators=(",", ":"))
    OUT.write_text(
        "// The data of the illustration units-and-forms (see illustration.js), written by\n"
        "// ../compute/compute.py. Values are in units of $100,000.\n"
        f"window.SHAPIQ_UNITS_AND_FORMS = {text};\n"
    )
    print(f"R² {r2:.3f}, prediction {prediction:.3f}, baseline {table[()]:.3f}, wrote {OUT}")
    print(
        "sum of Shapley values",
        sum(out["shapley"]),
        "= prediction - baseline",
        prediction - table[()],
    )


def _round(value: object) -> object:
    """Round floats to 3 decimals, recursively, to keep the file small."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


if __name__ == "__main__":
    main()
