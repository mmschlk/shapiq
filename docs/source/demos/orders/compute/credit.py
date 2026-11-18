"""Explain an XGBoost model of credit card default at every order: ``results/applicant(s).json``.

The data is "Default of credit card clients" (Yeh and Lien, 2009, UCI Machine Learning
Repository, CC BY 4.0): 30,000 credit card holders of a bank in Taiwan in 2005, 6,636 of whom
defaulted on their payment of the next month. The UCI site is not reachable from every machine,
so the script downloads the UCI file from a mirror on GitHub, pinned to a commit, and checks it
against the UCI file's hash, its size and its counts of rows, columns and defaults.

The model is an XGBoost classifier of depth 5 (300 trees, like ``housing.py``'s) trained on 80 %
of the clients, stratified by default. Its players are 8 of the 23 features of a client: the
credit limit, the age, the repayment status of the last three months (September, August and July
2005: -2 for no use of the card, -1 for paid in full, 0 for revolving credit, and from 1 on, the
months of delay), the last bill, and the last two payments (in September and in August). The model
predicts the log-odds of default, which is what shapiq explains. An absent feature takes its values
from 100 training clients (interventional explanations, as in background SHAP), and
``shapiq.TreeExplainer`` computes the Möbius transform of this game exactly from the trees, with
the Woodelf algorithm for many applicants at once. Every path of a tree of depth 5 splits on at
most 5 features, so FSII of order 5 is already the full decomposition, and exact.

Every order is scored as in ``housing.py``: by its R^2 beyond size under the Shapley distribution,
exactly, on all 254 coalitions but the empty and the full one. Two files:

- ``results/applicant.json``: one applicant of the test set, picked by a rule. Of the test
  applicants who defaulted and whom the model gives a probability of default above 0.7, and whose
  Shapley values fit better than their sizes alone (R^2 above 0), it is the one whose fit gains
  the most from the Shapley values to order 2: test applicant 4650, a bill of nearly all of the
  credit limit, almost nothing paid in August, and two months late (R^2 = 0.24 at order 1, 0.98 at
  order 2);
- ``results/applicants.json``: 2,000 random test applicants, with their features, their predicted
  probability of default, the R^2 of their Shapley values (the demo places them by these two) and
  their explanations, with the mean R^2 of every order, and the R^2 of the Shapley values of all
  test applicants.

Usage::

    python credit.py

The data is downloaded to ``results/credit/``. The explanations of all 6,000 test applicants take
a few seconds on four CPU cores, their FSII of every order and R^2 about a minute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from faithfulness import all_coalitions  # in this folder
from housing import explain  # in this folder: FSII of every order, with its R^2
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import shapiq

HERE = Path(__file__).resolve().parent
# the UCI file "default of credit card clients.xls", from a mirror on GitHub, at a commit; the same
# file is in other repositories (robertofranceschi/Default-Credit-Card-Prediction, dataset/)
URL = (
    "https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/"
    "585f77c33dff5a14dac8e5396bec24820d3db2f8/dataset/default%20of%20credit%20card%20clients.xls"
)
SHA256 = "c76095b2a65c6e1b0ef6bdfab785346065b02f3f50cab6751d5a46b4c01c9e44"
SIZE = 5_539_328  # bytes
CLIENTS, DEFAULTS = 30_000, 6_636  # as UCI and Yeh and Lien (2009) count them
DEPTH = 5  # the highest order of interaction: FSII of order 5 is exact
N_BACKGROUND = 100  # the training clients whose values absent features take
N_APPLICANTS = 2000  # the applicants of the population
SURE = 0.7  # the probability of default above which the model is sure of the applicant's
TARGET = "default payment next month"
# the features, as the data names them, and as the demo does
COLUMNS = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"]
NAMES = ["Credit limit", "Age", "Status Sep", "Status Aug", "Status Jul", "Bill Sep", "Paid Sep"]
NAMES += ["Paid Aug"]


def load(folder: Path) -> pd.DataFrame:
    """Return the clients of the UCI file, downloaded to ``folder`` once, and checked."""
    path = folder / "default of credit card clients.xls"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:  # (a fixed URL)
            path.write_bytes(response.read())
    content = path.read_bytes()
    if len(content) != SIZE or hashlib.sha256(content).hexdigest() != SHA256:
        msg = f"{path} is not the UCI file (its size or its hash differ)"
        raise ValueError(msg)
    data = pd.read_excel(path, header=1)  # (the first row is a second head: X1, ..., Y)
    if data.shape != (CLIENTS, 25) or int(data[TARGET].sum()) != DEFAULTS:
        msg = f"{path} has {data.shape} rows and columns, and {data[TARGET].sum()} defaults"
        raise ValueError(msg)
    return data


def main() -> None:
    """Train the model, explain the applicants, write ``results/applicant(s).json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    data = load(args.results / "credit")
    X = data[COLUMNS].to_numpy(dtype=float)
    y = data[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    model = XGBClassifier(
        n_estimators=300, max_depth=DEPTH, learning_rate=0.05, random_state=0, n_jobs=4
    ).fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    margin = model.predict(X_test, output_margin=True)  # the log-odds of default, explained
    auc, accuracy = roc_auc_score(y_test, probability), accuracy_score(y_test, probability > 0.5)
    print(f"test AUC {auc:.3f}, accuracy {accuracy:.3f} (defaults {y_test.mean():.1%})")

    rng = np.random.default_rng(0)
    background = X_train[rng.choice(len(X_train), N_BACKGROUND, replace=False)]
    explainer = shapiq.TreeExplainer(
        model,
        mode="interventional",
        reference_dataset=background,
        index="Moebius",
        max_order=X_train.shape[1],
    )
    moebius = explainer.explain_X(X_test)  # every test applicant, exactly
    n = X_test.shape[1]
    coalitions, weights = all_coalitions(n)

    # every test applicant's explanations of every order, and the fit of its Shapley values
    explained_all = [explain(moebius[i], coalitions, weights) for i in range(len(X_test))]
    shapley_r2 = np.array([orders[1]["r2"] for orders in explained_all])
    pair_r2 = np.array([orders[min(2, max(orders))]["r2"] for orders in explained_all])
    print("Shapley values' R^2 quartiles", np.quantile(shapley_r2, [0.25, 0.5, 0.75]).round(3))

    # one applicant: of the defaults the model is sure of, whose Shapley values fit better than
    # their sizes alone, the one whose fit gains the most at order 2
    caught = np.flatnonzero((y_test == 1) & (probability > SURE) & (shapley_r2 > 0))
    applicant = int(caught[np.argmax(pair_r2[caught] - shapley_r2[caught])])
    worse = np.mean(shapley_r2 < shapley_r2[applicant])
    print(f"applicant {applicant} of {len(caught)} sure defaults: p = ", end="")
    print(f"{probability[applicant]:.3f}, Shapley values' R^2 {shapley_r2[applicant]:.3f} ", end="")
    print(f"({worse:.1%} of the test applicants fit worse), order 2 {pair_r2[applicant]:.3f}")
    orders = explained_all[applicant]
    result = {
        "index": "FSII",
        "features": NAMES,
        "x": X_test[applicant].tolist(),
        "test_index": applicant,
        "defaulted": int(y_test[applicant]),
        "prediction": float(margin[applicant]),
        "probability": float(probability[applicant]),
        "baseline": float(moebius[applicant].baseline_value),
        "test_auc": float(auc),
        "test_accuracy": float(accuracy),
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
    (args.results / "applicant.json").write_text(json.dumps(result))
    print(f"applicant {applicant}: R^2 by order", [round(o["r2"], 3) for o in orders.values()])

    # many applicants: their explanations of every order
    applicants = np.random.default_rng(1).choice(len(X_test), N_APPLICANTS, replace=False)
    explained = [explained_all[int(i)] for i in applicants]
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
        "x": X_test[applicants].tolist(),
        "predictions": margin[applicants].tolist(),  # the log-odds of default
        "probabilities": probability[applicants].tolist(),
        "defaulted": y_test[applicants].tolist(),
        "shapley_r2": shapley_r2[applicants].tolist(),  # the demo places them by it
        "largest_order": largest,
        "n_players": n,
        "shapley_r2_quartiles": np.quantile(shapley_r2, [0.25, 0.5, 0.75]).tolist(),
        "orders": by_order,
    }
    (args.results / "applicants.json").write_text(json.dumps(result))
    print(
        f"{N_APPLICANTS} applicants: mean R^2 by order",
        [round(o["r2"], 3) for o in by_order.values()],
    )
    shares = [np.mean(shapley_r2 >= bound) for bound in (0.8, 0.9, 0.95)]
    print("test applicants whose Shapley values reach R^2 0.8, 0.9, 0.95:", np.round(shares, 3))


if __name__ == "__main__":
    main()
