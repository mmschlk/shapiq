"""Compute the data of the illustration "lab-stepper" (chapter 2.1 of the user guide).

One passenger of the Titanic, a gradient boosting classifier for the probability that a passenger
survived, on five features (sex, age, passenger class, fare, family aboard), and the explanations
that the illustration shows for her, all in percentage points (pp) of the probability of survival:

- the explanation game of the passenger with marginal masking: her predicted probability for every
  subset of the features, with the other features drawn from the training passengers, for all
  2^5 = 32 subsets;
- numbers: the Shapley values of the features and the pairwise Shapley interactions (k-SII of
  order 2), with shapiq's ``ExactComputer``, and the Shapley values of the units of a partition of
  the features, each unit one player (``partition_values``): who she is (sex and age), her ticket
  (class and fare), and her family aboard;
- curves: her predicted probability as one feature varies (her ICE curve), as the two features of a
  group vary (one curve per value of the first, a category), and the part of the probability as two
  features vary that neither feature explains alone (the pure interaction), on grids that pass
  through her own values, so that she lies on her curves.

Run it from the repository's root with the docs' dependencies installed; it downloads the data from
OpenML once (scikit-learn caches it in ``~/scikit_learn_data``) and writes ``../assets/data.js``::

    python docs/source/user_guide/illustrations/lab-stepper/compute/compute.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import shapiq

# the features: column of the prepared data, label, short label, description, and the grid of its
# values for the curves: the categories, or evenly spaced numbers (from, to, points)
FEATURES = [
    ("male", "Sex", "Sex", "female or male", {"categories": ["female", "male"]}),
    ("age", "Age", "Age", "in years", {"range": (1, 65, 33)}),
    ("pclass", "Class", "Class", "passenger class", {"categories": ["1st", "2nd", "3rd"]}),
    ("fare", "Fare", "Fare", "the ticket's price, in pounds", {"range": (0, 120, 31)}),
    ("family", "Family", "Family", "siblings, spouses, parents, and children aboard", {"steps": 8}),
]
# the units of "groups": who she is (sex and age), her ticket (class and fare), and her family
GROUPS = [((0, 1), "who she is"), ((2, 3), "her ticket"), ((4,), "her family")]
PASSENGER = "Ford, Mrs. Edward (Margaret Ann Watson)"
PURE_POINTS = 13  # points per numeric axis of a pure interaction surface
SEED = 0
OUT = Path(__file__).resolve().parents[1] / "assets" / "data.js"


def main() -> None:
    """Train the model, explain the passenger, and write the data."""
    frame = fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff").frame
    frame = frame.assign(
        male=(frame["sex"] == "male").astype(float),
        family=frame["sibsp"] + frame["parch"],
        survived=(frame["survived"].astype(str) == "1").astype(int),
    )
    frame = frame.dropna(subset=["age", "fare"]).reset_index(drop=True)
    columns = [column for column, *_ in FEATURES]
    x_all = frame[columns].to_numpy(float)
    y_all = frame["survived"].to_numpy()
    n = len(FEATURES)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.25, random_state=SEED, stratify=y_all
    )
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, min_samples_leaf=30, random_state=SEED
    ).fit(x_train, y_train)
    held_out = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, held_out)
    accuracy = accuracy_score(y_test, held_out > 0.5)

    row = int(np.flatnonzero(frame["name"] == PASSENGER)[0])
    passenger = x_all[row]

    def predict(rows: np.ndarray) -> np.ndarray:
        """The predicted probability of survival, in percent."""
        return 100 * model.predict_proba(rows)[:, 1]

    prediction = float(predict(passenger[None])[0])
    background = x_train  # marginal masking over all training passengers

    def game(coalitions: np.ndarray) -> np.ndarray:
        """Her prediction with the features of each coalition, the others from the background."""
        values = np.empty(len(coalitions))
        for k, coalition in enumerate(coalitions.astype(bool)):
            masked = background.copy()
            masked[:, coalition] = passenger[coalition]
            values[k] = predict(masked).mean()
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

    def grid(feature: int, points: int | None = None) -> np.ndarray:
        """The values of a feature for its curves: its categories, or evenly spaced numbers."""
        spec = FEATURES[feature][4]
        if "categories" in spec:
            return np.unique(x_all[:, feature])
        if "steps" in spec:
            return np.arange(spec["steps"], dtype=float)
        low, high, count = spec["range"]
        return np.linspace(low, high, points or count)

    def predict_with(changes: dict[int, np.ndarray]) -> np.ndarray:
        """Her prediction with the features of ``changes`` set to their (broadcast) values."""
        shape = np.broadcast_shapes(*(v.shape for v in changes.values()))
        rows = np.repeat(passenger[None], math.prod(shape), axis=0)
        for feature, values in changes.items():
            rows[:, feature] = np.broadcast_to(values, shape).ravel()
        return predict(rows).reshape(shape)

    # her ICE curves: the grid, with her own value added where it is missing, so that she lies on it
    curves = []
    for i in range(n):
        z = np.union1d(grid(i), passenger[i : i + 1])
        curves.append({"x": z.tolist(), "y": predict_with({i: z}).tolist()})

    # the curves of a group of two: one curve over the second feature per value of the first
    group_curves = {}
    for ids, _ in GROUPS:
        if len(ids) == 2:
            i, j = ids
            zi, zj = grid(i), np.union1d(grid(j), passenger[j : j + 1])
            surface = predict_with({i: zi[:, None], j: zj[None, :]})
            group_curves[f"{i}-{j}"] = {"by": zi.tolist(), "x": zj.tolist(), "y": surface.tolist()}

    # the pure interaction of each pair: the surface minus what each feature does alone
    pure = {}
    for i, j in itertools.combinations(range(n), 2):
        zi, zj = grid(i, PURE_POINTS), grid(j, PURE_POINTS)
        both = predict_with({i: zi[:, None], j: zj[None, :]})  # rows: feature i, columns: j
        only_i = predict_with({i: zi})[:, None]
        only_j = predict_with({j: zj})[None, :]
        pure[f"{i}-{j}"] = {
            "x": zi.tolist(),
            "y": zj.tolist(),
            "z": (both - only_i - only_j + prediction).tolist(),
        }

    numbers = {
        "prediction": prediction,
        "baseline": float(table[()]),
        "shapley": [float(shapley[(i,)]) for i in range(n)],
        "group_shapley": partition_values([ids for ids, _ in GROUPS]),
        "interactions": {
            f"{i}-{j}": float(ksii[(i, j)]) for i, j in itertools.combinations(range(n), 2)
        },
    }
    out = {
        "source": "Titanic passengers (OpenML 40945); GradientBoostingClassifier on 5 features",
        "passengers": len(frame),
        "train": len(x_train),
        "test": len(x_test),
        "auc": round(auc, 3),
        "accuracy": round(accuracy, 3),
        "passenger": {"name": PASSENGER, "survived": bool(y_all[row])},
        "features": [
            {
                "name": column,
                "label": label,
                "short": short,
                "description": description,
                "categories": spec.get("categories"),
                "value": float(passenger[i]),
            }
            for i, (column, label, short, description, spec) in enumerate(FEATURES)
        ],
        "groups": [{"ids": list(ids), "label": label} for ids, label in GROUPS],
        **_round(numbers, 3),
        **_round({"curves": curves, "group_curves": group_curves, "pure": pure}, 2),
    }
    text = json.dumps(out, separators=(",", ":"))
    OUT.write_text(
        "// The data of the illustration lab-stepper (see illustration.js), written by\n"
        "// ../compute/compute.py. Probabilities in percent, explanations in percentage points.\n"
        f"window.SHAPIQ_LAB_STEPPER = {text};\n"
    )
    print(f"{len(frame)} passengers, held-out AUC {auc:.3f}, accuracy {accuracy:.3f}")
    print(f"passenger: {PASSENGER} (row {row}), survived: {bool(y_all[row])}")
    print(f"prediction {prediction:.2f}%, average {table[()]:.2f}%")
    print("Shapley values (pp)", np.round(numbers["shapley"], 2).tolist())
    print("group Shapley values (pp)", np.round(numbers["group_shapley"], 2).tolist())
    print("k-SII of pairs (pp)", {k: round(v, 2) for k, v in numbers["interactions"].items()})
    print(
        f"sums: Shapley {sum(numbers['shapley']):.3f}, groups {sum(numbers['group_shapley']):.3f},",
        f"prediction - average {prediction - table[()]:.3f}",
    )
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
