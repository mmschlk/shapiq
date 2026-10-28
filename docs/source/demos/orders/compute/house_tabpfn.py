"""Explain TabPFN's prediction of the house of ``results/house.json`` at every order.

The house is the one of ``housing.py``, in Los Angeles: the same split of the California housing
dataset of scikit-learn, the same test house (of the test houses within 0.15 degrees of downtown
Los Angeles, the one whose Shapley values of the XGBoost model of depth 5 fit closest to
R^2 = 0.3). The model is TabPFN, a transformer pretrained on synthetic tables, which predicts by
in-context learning: it is not trained on the houses, but reads training houses (its context)
together with the house it predicts. Its context is 2,000 training houses drawn at random (with a
fixed seed), not all 16,512: on four CPU cores, TabPFN takes about six seconds per prediction with
this context.

TabPFN is a black box: unlike trees, it has no structure that would give its interactions exactly,
and no depth that would limit their order. But a house has only 8 features, so the game has only
2^8 = 256 coalitions: ``shapiq.ExactComputer`` evaluates all of them and computes the Möbius
transform exactly. The game is shapiq's ``TabPFNImputer``, the remove-and-contextualize paradigm
(Rundel et al., 2024): a feature absent from a coalition is not imputed, but removed, from the
house and from every house of TabPFN's context, and TabPFN predicts the house from the features
of the coalition alone. The empty coalition has no feature to predict from: its value is TabPFN's
mean prediction of the test houses (the imputer's empty prediction), the explanation's baseline.

As for the XGBoost model, ``shapiq`` converts the Möbius transform into k-SII of every order, and
scores every order by its R^2 on all 2^8 coalitions of the game (every feature present with
probability 1/2): how much of TabPFN's output on those coalitions the explanation reproduces. One
file, ``results/house_tabpfn.json``, in the format of ``results/house.json``, with the number of
training houses in TabPFN's context.

Usage::

    python house_tabpfn.py

The dataset is downloaded by scikit-learn to ``results/sklearn/``, TabPFN's weights from Hugging
Face on the first run. TabPFN's predictions of all 4,128 test houses (for its R^2 on them) take
about a minute on four CPU cores, its predictions of the 255 coalitions of the house about 25
minutes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from housing import LOS_ANGELES, NAMES, explain  # the XGBoost model's script, in this folder
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import shapiq

HERE = Path(__file__).resolve().parent
HOUSE = 837  # the test house of results/house.json, which housing.py picks (XGBoost of depth 5)
N_CONTEXT = 2000  # the training houses in TabPFN's context


def tabpfn(x_context: np.ndarray, y_context: np.ndarray) -> TabPFNRegressor:
    """Return TabPFN with the houses of its context: the model to explain."""
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,  # the weights that download without a license key
        device="cpu",
        random_state=0,
        ignore_pretraining_limits=True,  # a context of more than 1,000 houses on a CPU
    )
    return model.fit(x_context, y_context)


def moebius_transform(
    model: TabPFNRegressor,
    x_context: np.ndarray,
    y_context: np.ndarray,
    house: np.ndarray,
    empty_prediction: float,
) -> shapiq.InteractionValues:
    """Return the Möbius transform of the house's remove-and-contextualize game, exactly."""
    game = shapiq.TabPFNImputer(
        model,
        x_context,
        y_context,
        empty_prediction=empty_prediction,
        predict_function=lambda model, x: model.predict(x),
    ).fit(house)
    return shapiq.ExactComputer(game)("Moebius")  # all 2^8 coalitions


def main() -> None:
    """Explain TabPFN's prediction of the house and write ``results/house_tabpfn.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    data = fetch_california_housing(data_home=str(args.results / "sklearn"))
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=0
    )
    house = X_test[HOUSE]
    print(f"house {HOUSE} at latitude {house[6]}, longitude {house[7]}")
    assert np.all(np.abs(house[6:] - LOS_ANGELES) < 0.15)  # noqa: S101  (in Los Angeles)
    context = np.random.default_rng(0).choice(len(X_train), N_CONTEXT, replace=False)
    x_context, y_context = X_train[context], y_train[context]
    model = tabpfn(x_context, y_context)

    start = time.time()
    test_predictions = model.predict(X_test)
    test_r2 = float(r2_score(y_test, test_predictions))
    prediction = float(model.predict(house[None])[0])
    print(f"test R^2 {test_r2:.3f} ({time.time() - start:.0f} s); prediction {prediction:.3f}")

    # the game on all coalitions, its empty coalition TabPFN's mean prediction of the test houses
    start = time.time()
    moebius = moebius_transform(
        model, x_context, y_context, house, float(np.mean(test_predictions))
    )
    print(f"255 coalitions in {time.time() - start:.0f} s")
    full = moebius.baseline_value + sum(moebius[key] for key in moebius.interaction_lookup if key)
    assert np.isclose(full, prediction)  # noqa: S101  (the full decomposition is exact)

    n = X_test.shape[1]
    coalitions = np.array(list(itertools.product([False, True], repeat=n)))
    orders = explain(moebius, coalitions)
    result = {
        "features": NAMES,
        "x": house.tolist(),
        "prediction": prediction,
        "baseline": float(moebius.baseline_value),
        "test_r2": test_r2,
        "context": N_CONTEXT,
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
    (args.results / "house_tabpfn.json").write_text(json.dumps(result))
    print(f"house {HOUSE}: R^2 by order", [round(o["r2"], 3) for o in orders.values()])


if __name__ == "__main__":
    main()
