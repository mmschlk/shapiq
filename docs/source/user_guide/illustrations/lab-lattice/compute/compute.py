"""Compute the data of the illustration "lab-lattice" (chapter 2.1 of the user guide).

One hour of the UCI bike sharing data (a working-day morning at 8:00), a gradient boosting model
that predicts the rentals of an hour from four features (hour, working day, temperature, humidity),
and the explanations that the illustration hangs on the sets of the lattice of all 2^4 = 16 subsets
of the features:

- the explanation game of the hour with marginal masking: its prediction for every subset of the
  features, with the other features drawn from background hours, for all 16 subsets;
- numbers, with shapiq's ``ExactComputer``: the Shapley values of the features, the Shapley
  interactions up to pairs (k-SII of order 2), and the Möbius transform of all 16 subsets (the
  empty set holds the average prediction); and the Shapley values of the two groups of a partition,
  time (hour, working day) and weather (temperature, humidity), each group one player;
- curves: the hour's prediction as the features of a set vary, the others as they are (ICE curves of
  the single features and ICE surfaces of the two groups), and the components f_S of the functional
  decomposition of the model with marginal masking, f = sum of f_S over all S, of every subset S,
  whose value at the hour's own features is its Möbius value. A component of three or more
  features is a slice: over two of its features, the others at the hour's values.

The grids pass through the hour's own values, so that the hour lies on every curve and surface.
Displayed numbers are whole rentals, rounded so that the parts add up to the shown total.

The data is downloaded once from the UCI repository into scikit-learn's data folder
(``sklearn.datasets.get_data_home()``). Run it from the repository's root with the docs'
dependencies installed; it writes ``../assets/data.js``::

    python docs/source/user_guide/illustrations/lab-lattice/compute/compute.py
"""

from __future__ import annotations

import itertools
import json
import math
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import get_data_home
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import shapiq

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
CACHE = Path(get_data_home()) / "uci_bike_sharing" / "bike+sharing+dataset.zip"
FEATURES = [  # column, label, scale to the shown unit, unit
    ("hr", "Hour", 1, ":00"),
    ("workingday", "Working day", 1, ""),
    ("temp", "Temperature", 41, "°C"),  # the data divides degrees Celsius by 41
    ("hum", "Humidity", 100, "%"),  # the data divides percent by 100
]
BINARY = {1}  # the features with two values, 0 and 1
DAY, HOUR = "2011-05-19", 8  # the explained hour: a humid working-day morning, a Thursday
GROUPS = [(0, 1), (2, 3)]  # the partition: time and weather
GROUP_NAMES = ["Time", "Weather"]
BACKGROUND = 1000  # background hours for marginal masking
GRID_LINE = 32  # points of a curve over temperature or humidity
GRID_MAP = 16  # points per axis of temperature or humidity on a surface
QUANTILES = (0.01, 0.99)  # the range of a grid
SEED = 0
OUT = Path(__file__).resolve().parents[1] / "assets" / "data.js"

Subset = tuple[int, ...]


def load() -> pd.DataFrame:
    """The hourly bike sharing data, downloaded once into the cache."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            CACHE.write_bytes(response.read())
    with zipfile.ZipFile(CACHE) as archive, archive.open("hour.csv") as file:
        return pd.read_csv(file)


def main() -> None:
    """Train the model, explain the hour, and write the data."""
    started = time.perf_counter()
    data = load()
    n = len(FEATURES)
    x_all = data[[column for column, *_ in FEATURES]].to_numpy(float)
    y_all = data["cnt"].to_numpy(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=SEED
    )
    model = HistGradientBoostingRegressor(max_iter=300, random_state=SEED).fit(x_train, y_train)
    r2 = r2_score(y_test, model.predict(x_test))
    row = data.index[(data["dteday"] == DAY) & (data["hr"] == HOUR)][0]
    x = x_all[row]
    prediction = float(model.predict(x[None])[0])

    rng = np.random.default_rng(SEED)
    background = x_train[rng.choice(len(x_train), BACKGROUND, replace=False)]

    # -- the game and the numbers -------------------------------------------------------------------

    def game(coalitions: np.ndarray) -> np.ndarray:
        """Predict the hour with the features of each coalition, the others from the background."""
        values = np.empty(len(coalitions))
        for k, coalition in enumerate(coalitions.astype(bool)):
            masked = background.copy()
            masked[:, coalition] = x[coalition]
            values[k] = model.predict(masked).mean()
        return values

    subsets: list[Subset] = [
        s for size in range(n + 1) for s in itertools.combinations(range(n), size)
    ]
    coalitions = np.array([[i in s for i in range(n)] for s in subsets])
    table = dict(zip(subsets, game(coalitions), strict=True))
    baseline = table[()]

    computer = shapiq.ExactComputer(game=game, n_players=n)
    shapley = computer("SV", order=1)
    ksii = computer("k-SII", order=2)
    moebius = computer("Moebius", order=n)

    def group_values(parts: list[Subset]) -> list[float]:
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

    singles = [(i,) for i in range(n)]
    pairs = list(itertools.combinations(range(n), 2))
    shown_total = round(prediction) - round(baseline)
    explanations = {
        "features": dict(zip(singles, [shapley[s] for s in singles], strict=True)),
        "groups": dict(zip(GROUPS, group_values(GROUPS), strict=True)),
        "pairs": {s: ksii[s] for s in singles + pairs},
        "all": {s: moebius[s] for s in subsets if s},
    }
    numbers = {}
    for name, values in explanations.items():
        shown = _round_to(list(values.values()), shown_total)
        numbers[name] = {
            _key(s): [round(float(v), 2), w]
            for (s, v), w in zip(values.items(), shown, strict=True)
        }
    numbers["all"][_key(())] = [round(float(baseline), 2), round(baseline)]

    # -- the curves -----------------------------------------------------------------------------------

    def grid(feature: int, points: int) -> np.ndarray:
        """The values of a feature on a curve or surface.

        All hours, both values of a binary feature, or evenly spaced values over the feature's
        range, one of them the hour's own value.
        """
        if feature == 0:
            return np.arange(24.0)
        if feature in BINARY:
            return np.array([0.0, 1.0])
        low, high = np.quantile(x_all[:, feature], QUANTILES)
        step = (high - low) / (points - 1)
        at = int(np.clip(round((x[feature] - low) / step), 0, points - 1))
        return x[feature] + (np.arange(points) - at) * step

    def axes_of(s: Subset) -> Subset:
        """The features that the curve or surface of a set runs over.

        All of them for one or two features, else the first two that are not binary; the others
        stay at the hour's values (a slice).
        """
        if len(s) <= 2:
            return s
        return tuple(i for i in s if i not in BINARY)[:2]

    def grids_of(axes: Subset) -> list[np.ndarray]:
        """The grids of a curve, a pair of curves (with a binary feature), or a surface."""
        surface = len(axes) == 2 and not any(i in BINARY for i in axes)
        return [grid(i, GRID_MAP if surface else GRID_LINE) for i in axes]

    def frame(axes: Subset, grids: list[np.ndarray], features: Subset) -> list[np.ndarray]:
        """For each feature of ``features``, its values on the grid (broadcast), or the hour's."""
        shape = tuple(len(g) for g in grids)
        out = []
        for i in features:
            if i in axes:
                k = axes.index(i)
                out.append(
                    np.broadcast_to(
                        grids[k].reshape([-1 if a == k else 1 for a in range(len(axes))]), shape
                    )
                )
            else:
                out.append(np.full(shape, x[i]))
        return out

    def ice(s: Subset) -> np.ndarray:
        """The hour's prediction as the features of ``s`` vary on their grids, the others as they are."""
        axes = axes_of(s)
        grids = grids_of(axes)
        values = frame(axes, grids, s)
        rows = np.repeat(x[None], values[0].size, axis=0)
        for i, v in zip(s, values, strict=True):
            rows[:, i] = v.ravel()
        return model.predict(rows).reshape(values[0].shape)

    cache: dict[tuple, np.ndarray] = {}

    def partial_dependence(t: Subset, axes: Subset, grids: list[np.ndarray]) -> np.ndarray:
        """The partial dependence of ``t`` on the grid of ``axes``: the game at every point.

        The average prediction over the background, with the features of ``t`` set to their values
        on the grid, or to the hour's where they are not an axis.
        """
        shape = tuple(len(g) for g in grids)
        key = (t, tuple((i, axes.index(i) if i in axes else -1) for i in t), shape)
        if key not in cache:
            values = frame(axes, grids, t)
            points = math.prod(shape)
            rows = np.tile(background, (points, 1))
            for i, v in zip(t, values, strict=True):
                rows[:, i] = np.repeat(v.ravel(), len(background))
            cache[key] = model.predict(rows).reshape(points, len(background)).mean(1).reshape(shape)
        return cache[key]

    def component(s: Subset) -> np.ndarray:
        """The component f_S of the functional decomposition on the grid.

        The partial dependence of S minus that of all its subsets, by inclusion and exclusion (the
        Möbius transform of the partial dependences).
        """
        axes = axes_of(s)
        grids = grids_of(axes)
        total = np.zeros(tuple(len(g) for g in grids))
        for size in range(len(s) + 1):
            for t in itertools.combinations(s, size):
                total = total + (-1) ** (len(s) - size) * partial_dependence(t, axes, grids)
        return total

    def curve(s: Subset, values: np.ndarray) -> dict:
        """A curve or surface for the data.

        Its axes, the features fixed at the hour's values, the grids in shown units, the index of
        the hour on each grid, and the values.
        """
        axes = axes_of(s)
        grids = grids_of(axes)
        at = [int(np.argmin(np.abs(g - x[i]))) for i, g in zip(axes, grids, strict=True)]
        return {
            "axes": list(axes),
            "fixed": [i for i in s if i not in axes],
            "grids": [
                [round(float(v) * FEATURES[i][2], 1) for v in g]
                for i, g in zip(axes, grids, strict=True)
            ],
            "at": at,
            "z": np.round(values, 1).tolist(),
        }

    curves = {
        "ice": {_key(s): curve(s, ice(s)) for s in singles + GROUPS},
        "components": {_key(s): curve(s, component(s)) for s in subsets if s},
    }
    curves["components"][_key(())] = {
        "axes": [],
        "fixed": [],
        "grids": [],
        "at": [],
        "z": round(baseline, 1),
    }

    # the check: each component at the hour's own features is its Möbius value
    deviation = max(
        abs(component(s)[tuple(curves["components"][_key(s)]["at"])] - moebius[s])
        for s in subsets
        if s
    )

    out = {
        "source": "UCI bike sharing, hourly; HistGradientBoostingRegressor on 4 features",
        "r2": round(r2, 2),
        "background": BACKGROUND,
        "hour": {"day": DAY, "weekday": pd.Timestamp(DAY).day_name(), "actual": int(y_all[row])},
        "features": [
            {"label": label, "value": round(float(x[i]) * scale, 1), "unit": unit}
            for i, (_, label, scale, unit) in enumerate(FEATURES)
        ],
        "binary": sorted(BINARY),
        "groups": [list(g) for g in GROUPS],
        "groupNames": GROUP_NAMES,
        "prediction": [round(prediction, 2), round(prediction)],
        "baseline": [round(float(baseline), 2), round(baseline)],
        "numbers": numbers,
        "curves": curves,
    }
    text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    OUT.write_text(
        "// The data of the illustration lab-lattice (see illustration.js), written by\n"
        "// ../compute/compute.py. Values are rentals per hour.\n"
        f"window.SHAPIQ_LAB_LATTICE = {text};\n",
        encoding="utf-8",
    )

    print(
        f"R² {r2:.3f} on held-out hours; {DAY} {HOUR}:00, {len(OUT.read_bytes()) / 1000:.1f} kB to {OUT}"
    )
    print(f"prediction {prediction:.2f} (actual {int(y_all[row])}), average {baseline:.2f}")
    for name, values in numbers.items():
        shown = sum(w for _, w in values.values())
        print(
            f"{name:9s} shown parts add up to {shown} (raw {sum(v for v, _ in values.values()):.2f})"
        )
    print(
        f"components at the hour minus Möbius values: at most {deviation:.2e} "
        f"(the largest Möbius value: {max(abs(moebius[s]) for s in subsets if s):.1f})"
    )
    print(f"done in {time.perf_counter() - started:.0f} s")


def _key(s: Subset) -> str:
    """A set of features as a key of the data: its 0-based indices, "" for the empty set."""
    return "".join(str(i) for i in s)


def _round_to(values: list[float], total: int) -> list[int]:
    """Whole numbers near ``values`` that add up to ``total``.

    Each value is rounded down, and the rest is given to those that lost the most (largest
    remainders).
    """
    shown = [math.floor(v) for v in values]
    order = sorted(range(len(values)), key=lambda k: values[k] - shown[k], reverse=True)
    rest = total - sum(shown)
    for j in range(abs(rest)):
        k = order[j % len(order)] if rest > 0 else order[-1 - j % len(order)]
        shown[k] += 1 if rest > 0 else -1
    return shown


if __name__ == "__main__":
    main()
