"""Script to compute explanations used in the tests of the ``shapiq`` library.

A script which uses the ExactComputer to compute a variety of explanations for a couple of
model and datasets to be used for testing in the ``shapiq`` library.
"""

# flake8: noqa: T201

from __future__ import annotations

from pathlib import Path
from typing import get_args

from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.game_theory.exact import ExactComputer
from tests.fixtures.data import get_california_housing_train_test_explain
from tests.fixtures.games import get_california_housing_imputer
from tests.fixtures.models import get_california_housing_random_forest

if __name__ == "__main__":
    RANDOM_SEED = 42
    SAVE_PATH = Path(__file__).parent.parent / "data" / "interaction_values" / "california_housing"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Creating and saving interaction values to {SAVE_PATH}")

    # get the dataset and train a model
    x_train, y_train, x_test, y_test, x_explain = get_california_housing_train_test_explain()
    model = get_california_housing_random_forest()
    print(f"Model score: {model.score(x_test, y_test)}")
    print(f"Model prediction for x_explain: {model.predict(x_explain)}")

    # compute explanations
    imputer = get_california_housing_imputer()
    imputer_hash = hash(
        (
            imputer.sample_size,
            imputer.joint_marginal_distribution,
            imputer.normalize,
            imputer.random_state,
        )
    )
    print("Imputer hash:", imputer_hash)
    imputer.verbose = True
    exact_computer = ExactComputer(game=imputer, n_players=imputer.n_players, evaluate_game=True)
    save_name = f"iv_california_housing_imputer_{imputer_hash}"
    value_indices = ["SV", "BV"]
    for index in value_indices:
        iv = exact_computer(index=index, order=1)
        iv = iv.get_n_order(order=1)
        iv.save(path=SAVE_PATH / f"{save_name}_index={index}_order=1.pkl")
        print(f"Interaction values for index {index} (order 1):")
        print(iv)

    # compute Moebius as well
    iv = exact_computer(index="Moebius", order=imputer.n_players)
    iv.save(path=SAVE_PATH / f"{save_name}_index=Moebius_order={imputer.n_players}.pkl")
    print("Moebius interaction values:")
    print(iv)

    # compute interaction values for all indices that are in the ExplainerIndices
    interaction_indices = list(get_args(ExplainerIndices))
    orders = list(range(1, imputer.n_players + 1))
    for index in interaction_indices:
        if index in value_indices:
            continue
        for order in orders:
            iv = exact_computer(index=index, order=order)
            iv = iv.get_n_order(min_order=1, max_order=order)
            iv.save(path=SAVE_PATH / f"{save_name}_index={index}_order={order}.pkl")
            print(f"Interaction values for index {index} (order {order}):")
            print(iv)
