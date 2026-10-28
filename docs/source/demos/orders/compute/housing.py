"""Explain an XGBoost model of California housing at every order: ``results/house(s).json``.

The model is an XGBoost regressor of depth 5 (300 trees) trained on 80 % of the California housing
dataset of scikit-learn. Its players are the 8 features of a house. An absent feature takes its
values from 100 training houses (interventional explanations, as in background SHAP), and
``shapiq.TreeExplainer`` computes the Möbius transform of this game exactly from the trees, with
the Woodelf algorithm for many houses at once. Every path of a tree of depth 5 splits on at most 5
features, so the Möbius transform has no interaction of more than 5 features: FSII of order 5 is
already the full decomposition (FSII of order n = 8), and exact.

``shapiq`` converts the Möbius transform of every house into FSII of every order, and scores every
order by its R^2 under the Shapley distribution (``faithfulness.py``), exactly: on all 254
coalitions of the house's game but the empty and the full one, each weighted by its probability.
It is how much of the model's output on those coalitions the explanation reproduces. The empty
coalition's value, the explanation's baseline, is the model's mean prediction of the 100 training
houses, part of the game and of the explanation alike. Two files:

- ``results/house.json``: one house in Los Angeles, test house 837, where latitude and longitude
  interact strongly: its Shapley values fit worse than those of 96 % of the test houses within
  0.15 degrees of downtown Los Angeles (R^2 = 0.39);
- ``results/houses.json``: 150 random test houses, for a heatmap of their explanations, with the
  mean R^2 of every order, and the R^2 of the Shapley values of all test houses.

Usage::

    python housing.py

The dataset is downloaded by scikit-learn to ``results/sklearn/``. The explanations of all 4,128
test houses take a few seconds on four CPU cores, their FSII of every order and R^2 about 40
seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from faithfulness import all_coalitions, fsii, predictions, r2  # in this folder
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import shapiq
from shapiq.game_theory.moebius_converter import MoebiusConverter

HERE = Path(__file__).resolve().parent
DEPTH = 5  # the highest order of interaction: FSII of order 5 is exact
N_BACKGROUND = 100  # the training houses whose values absent features take
N_HOUSES = 150  # the houses of the heatmap
LOS_ANGELES = (34.05, -118.25)  # downtown, as latitude and longitude
HOUSE = 837  # the test house of house.json, in Los Angeles, whose Shapley values fit poorly
NAMES = ["Income", "House age", "Rooms", "Bedrooms", "Population", "Occupancy", "Latitude"]
NAMES += ["Longitude"]  # the features of scikit-learn's California housing, in its order


def explain(
    moebius: shapiq.InteractionValues, coalitions: np.ndarray, weights: np.ndarray
) -> dict[int, dict]:
    """Return FSII of every order of a house, with its R^2 on the coalitions of its game.

    The coalitions are weighted by their probability under the Shapley distribution.
    """
    game = predictions({key: moebius[key] for key in moebius.interaction_lookup}, coalitions)
    converter = MoebiusConverter(moebius)
    largest = max(len(key) for key in moebius.interaction_lookup if abs(moebius[key]) > 1e-9)
    orders = {}
    for order in range(1, largest + 1):
        values = fsii(converter, order)
        orders[order] = {
            "r2": r2(game, predictions(values, coalitions), weights),
            "values": {key: value for key, value in values.items() if key},
        }
    return orders


def main() -> None:
    """Train the model, explain the houses, and write ``results/house.json`` and ``houses.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    data = fetch_california_housing(data_home=str(args.results / "sklearn"))
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=0
    )
    model = XGBRegressor(
        n_estimators=300, max_depth=DEPTH, learning_rate=0.1, random_state=0, n_jobs=4
    ).fit(X_train, y_train)
    rng = np.random.default_rng(0)
    background = X_train[rng.choice(len(X_train), N_BACKGROUND, replace=False)]
    explainer = shapiq.TreeExplainer(
        model,
        mode="interventional",
        reference_dataset=background,
        index="Moebius",
        max_order=X_train.shape[1],
    )
    moebius = explainer.explain_X(X_test)  # every test house, exactly
    n = X_test.shape[1]
    coalitions, weights = all_coalitions(n)

    # the fit of the Shapley values of every test house
    shapley_r2 = np.array(
        [explain(moebius[i], coalitions, weights)[1]["r2"] for i in range(len(X_test))]
    )
    print(f"test R^2 {model.score(X_test, y_test):.3f}; Shapley values' R^2 quartiles", end=" ")
    print(np.quantile(shapley_r2, [0.25, 0.5, 0.75]).round(3))

    # one house in Los Angeles, where the Shapley values fit poorly
    house = HOUSE
    near = np.flatnonzero(
        (np.abs(X_test[:, 6] - LOS_ANGELES[0]) < 0.15)
        & (np.abs(X_test[:, 7] - LOS_ANGELES[1]) < 0.15)
    )
    assert house in near  # noqa: S101  (in Los Angeles)
    worse = np.mean(shapley_r2[near] < shapley_r2[house])
    print(f"house {house}: Shapley values' R^2 {shapley_r2[house]:.3f}, ", end="")
    print(f"{worse:.0%} of the {len(near)} test houses near downtown fit worse")
    orders = explain(moebius[house], coalitions, weights)
    result = {
        "index": "FSII",
        "features": NAMES,
        "x": X_test[house].tolist(),
        "prediction": float(model.predict(X_test[house : house + 1])[0]),
        "baseline": float(moebius[house].baseline_value),
        "test_r2": float(model.score(X_test, y_test)),
        "largest_order": max(orders),
        "n_players": n,
        "orders": {
            order: {
                "r2": explanation["r2"],
                "values": [[list(key), value] for key, value in explanation["values"].items()],
            }
            for order, explanation in orders.items()
        },
    }
    (args.results / "house.json").write_text(json.dumps(result))
    print(f"house {house}: R^2 by order", [round(o["r2"], 3) for o in orders.values()])

    # many houses: their explanations of every order, for the heatmap
    houses = np.random.default_rng(1).choice(len(X_test), N_HOUSES, replace=False)
    explained = [explain(moebius[int(i)], coalitions, weights) for i in houses]
    largest = max(max(orders) for orders in explained)
    by_order = {}
    for order in range(1, largest + 1):
        keys = sorted({key for e in explained for key in e[min(order, max(e))]["values"]})
        by_order[order] = {
            "r2": float(np.mean([e[min(order, max(e))]["r2"] for e in explained])),
            "values": {
                ",".join(map(str, key)): [
                    e[min(order, max(e))]["values"].get(key, 0.0) for e in explained
                ]
                for key in keys
            },
        }
    result = {
        "index": "FSII",
        "features": NAMES,
        "predictions": model.predict(X_test[houses]).tolist(),
        "largest_order": largest,
        "n_players": n,
        "shapley_r2_quartiles": np.quantile(shapley_r2, [0.25, 0.5, 0.75]).tolist(),
        "orders": by_order,
    }
    (args.results / "houses.json").write_text(json.dumps(result))
    print(f"{N_HOUSES} houses: mean R^2 by order", [round(o["r2"], 3) for o in by_order.values()])


if __name__ == "__main__":
    main()
