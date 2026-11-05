"""Compute the data of the illustration "lab-hulls" (chapter 2.1 of the user guide).

One patient of scikit-learn's diabetes data (in raw units), a gradient boosting model on six of its
features, and the explanations that the illustration draws on the units around the features:

- the explanation game of the patient with marginal masking: the model's average prediction when
  the features of a subset keep the patient's values and the others take those of all 442 patients
  of the data, for all 2^6 = 64 subsets;
- numbers: the Shapley values of the features and the pairwise Shapley interactions (k-SII of
  order 2), with shapiq's ``ExactComputer``, and the Shapley values of the units of a partition of
  the features, each unit one player (``partition_values``): BMI and blood pressure as one unit
  (the body), HDL and log triglycerides as one unit (the lipids), age and glucose alone;
- curves: the patient's prediction as one feature varies (its ICE curve), as the two features of a
  group vary (a surface), and, for each pair, the part of that surface that neither feature
  explains alone (the pure interaction), on evenly spaced grids that pass through the patient's own
  values, so that the patient lies on its curves and surfaces.

The data has 442 patients, few for a model: the script reports the model's R² on held-out patients
with 5-fold cross-validation, then fits the model on all patients and explains one of them. The
illustration shows values to one decimal; the Shapley values are rounded here so that they add up
to the shown prediction minus the shown average (largest remainders).

Run it from the repository's root with the docs' dependencies installed; it writes
``../assets/data.js``::

    python docs/source/user_guide/illustrations/lab-hulls/compute/compute.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

import shapiq

FEATURES = [  # name in the data, label, short label, unit, decimals shown
    ("age", "Age", "Age", "years", 0),
    ("bmi", "BMI", "BMI", "kg/m²", 1),
    ("bp", "Blood pressure", "BP", "mm Hg", 0),
    ("s6", "Glucose", "Glucose", "mg/dL", 0),
    ("s3", "HDL", "HDL", "mg/dL", 0),
    ("s5", "Log triglycerides", "Log TG", "", 2),
]
# the units of "groups": the body and the lipids as one unit each, age and glucose alone
PARTITION = [("age",), ("bmi", "bp"), ("s6",), ("s3", "s5")]
GROUP_NAMES = ["", "the body", "", "the lipids"]
PATIENT = 400  # the explained patient, row 400 of the data: the body raises, the lipids lower
GRID_1D = 40  # points of a curve
GRID_2D = 20  # points per axis of a surface
QUANTILES = (0.02, 0.98)  # the range of a feature's grid
SEED = 0
OUT = Path(__file__).resolve().parents[1] / "assets" / "data.js"


def make_model() -> GradientBoostingRegressor:
    """Shallow trees, a slow learning rate, and subsampling: a model for a small data set."""
    return GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=SEED,
    )


def shown(values: list[float], total: float) -> list[float]:
    """Values to one decimal that add up to ``total``, to one decimal: the largest remainders up."""
    tenths = [10 * v for v in values]
    floors = [math.floor(v) for v in tenths]
    order = sorted(range(len(values)), key=lambda k: floors[k] - tenths[k])
    rest = round(10 * total) - sum(floors)
    for k in order[: max(rest, 0)]:
        floors[k] += 1
    for k in order[::-1][: max(-rest, 0)]:
        floors[k] -= 1
    return [f / 10 for f in floors]


def main() -> None:
    """Train the model, explain the patient, and write the data."""
    data = load_diabetes(scaled=False)
    columns = [list(data.feature_names).index(name) for name, *_ in FEATURES]
    x_all, y_all = data.data[:, columns], data.target
    n = len(FEATURES)

    folds = KFold(5, shuffle=True, random_state=SEED)
    r2_folds = cross_val_score(make_model(), x_all, y_all, cv=folds, scoring="r2")
    model = make_model().fit(x_all, y_all)
    patient = x_all[PATIENT]
    prediction = float(model.predict(patient[None])[0])
    background = x_all

    def game(coalitions: np.ndarray) -> np.ndarray:
        """Predict the patient with the features of each coalition, the others from the data."""
        values = np.empty(len(coalitions))
        for row, coalition in enumerate(coalitions.astype(bool)):
            masked = background.copy()
            masked[:, coalition] = patient[coalition]
            values[row] = model.predict(masked).mean()
        return values

    subsets = [s for size in range(n + 1) for s in itertools.combinations(range(n), size)]
    coalitions = np.array([[i in s for i in range(n)] for s in subsets])
    table = dict(zip(subsets, game(coalitions), strict=True))

    computer = shapiq.ExactComputer(game=game, n_players=n)
    shapley = computer("SV", order=1)
    ksii = computer("k-SII", order=2)

    def partition_values(parts: list[tuple[int, ...]]) -> list[float]:
        """The Shapley values of the parts of a partition, in the game with each part one player."""
        m = len(parts)
        values = []
        for p, part in enumerate(parts):
            others = [q for q in range(m) if q != p]
            value = 0.0
            for size in range(m):
                weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
                for rest in itertools.combinations(others, size):
                    members = tuple(sorted(i for q in rest for i in parts[q]))
                    value += weight * (table[tuple(sorted(members + part))] - table[members])
            values.append(value)
        return values

    def grid(feature: int, points: int) -> np.ndarray:
        """Evenly spaced values over the feature's range, one of them the patient's own value."""
        low, high = np.quantile(x_all[:, feature], QUANTILES)
        step = (high - low) / (points - 1)
        at = int(np.clip(round((patient[feature] - low) / step), 0, points - 1))
        return patient[feature] + (np.arange(points) - at) * step

    def predict_with(changes: dict[int, np.ndarray]) -> np.ndarray:
        """Predict the patient with the features of ``changes`` set to their (broadcast) values."""
        shape = np.broadcast_shapes(*(v.shape for v in changes.values()))
        rows = np.repeat(patient[None], math.prod(shape), axis=0)
        for feature, values in changes.items():
            rows[:, feature] = np.broadcast_to(values, shape).ravel()
        return model.predict(rows).reshape(shape)

    curves = []
    for i in range(n):
        z = grid(i, GRID_1D)
        curves.append({"x": z.tolist(), "y": predict_with({i: z}).tolist()})

    names = [name for name, *_ in FEATURES]
    partition = [tuple(names.index(name) for name in part) for part in PARTITION]
    surfaces, pure = {}, {}
    for i, j in itertools.combinations(range(n), 2):
        zi, zj = grid(i, GRID_2D), grid(j, GRID_2D)
        both = predict_with({i: zi[:, None], j: zj[None, :]})  # rows: feature i, columns: j
        only_i = predict_with({i: zi})[:, None]
        only_j = predict_with({j: zj})[None, :]
        key = f"{i}-{j}"
        if (i, j) in partition:
            surfaces[key] = {"x": zi.tolist(), "y": zj.tolist(), "z": both.tolist()}
        pure[key] = (both - only_i - only_j + prediction).tolist()

    baseline = float(table[()])
    total = round(prediction, 1) - round(baseline, 1)
    shapley_values = [float(shapley[(i,)]) for i in range(n)]
    group_values = partition_values(partition)
    pairs = {f"{i}-{j}": float(ksii[(i, j)]) for i, j in itertools.combinations(range(n), 2)}
    out = {
        "source": "scikit-learn's diabetes data, raw units; GradientBoostingRegressor on 6 features",
        "patients": len(y_all),
        "patient": PATIENT,
        "r2": round(float(r2_folds.mean()), 2),
        "folds": len(r2_folds),
        "features": [  # the patient's values exactly, as in the data
            {
                "name": name,
                "label": label,
                "short": short,
                "unit": unit,
                "decimals": decimals,
                "value": round(float(patient[i]), 4),
            }
            for i, (name, label, short, unit, decimals) in enumerate(FEATURES)
        ],
        "partition": [list(part) for part in partition],
        "group_names": GROUP_NAMES,
        "prediction": round(prediction, 1),
        "baseline": round(baseline, 1),
        # the values, and those shown, which add up to the shown prediction minus the average
        "shapley": _round(shapley_values, 2),
        "shapley_shown": shown(shapley_values, total),
        "partition_shapley": _round(group_values, 2),
        "partition_shown": shown(group_values, total),
        "interactions": _round(pairs, 2),
        "curves": _round(curves, 1),
        "surfaces": _round(surfaces, 1),
        "pure": _round(pure, 2),
    }
    text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    OUT.write_text(
        "// The data of the illustration lab-hulls (see illustration.js), written by\n"
        "// ../compute/compute.py. Values are on the scale of the target: disease progression\n"
        "// one year after baseline.\n"
        f"window.SHAPIQ_LAB_HULLS = {text};\n",
        encoding="utf-8",
    )
    print(f"R² on held-out patients, {len(r2_folds)} folds: {np.round(r2_folds, 3)}")
    print(f"mean R² {r2_folds.mean():.3f}; patient {PATIENT}, observed {y_all[PATIENT]:.0f}")
    print(f"prediction {prediction:.2f}, average prediction {baseline:.2f}")
    print(f"Shapley values {np.round(shapley_values, 2)}, sum {sum(shapley_values):.2f}")
    print(f"group values {np.round(group_values, 2)}, sum {sum(group_values):.2f}")
    print(f"shown: {out['shapley_shown']} and {out['partition_shown']}, sum {total:.1f}")
    top = sorted(pairs.items(), key=lambda item: -abs(item[1]))[:4]
    print("strongest interactions", [(key, round(value, 2)) for key, value in top])
    print(f"wrote {OUT} ({OUT.stat().st_size / 1000:.1f} kB)")


def _round(value: object, digits: int) -> object:
    """Round floats to ``digits`` decimals, recursively, to keep the file small."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    return value


if __name__ == "__main__":
    main()
