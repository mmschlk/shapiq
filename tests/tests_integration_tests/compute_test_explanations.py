"""Script to compute explanations used in the tests of the ``shapiq`` library.

A script which uses the ExactComputer to compute a variety of explanations for a couple of
model and datasets to be used for testing in the ``shapiq`` library.
"""

# flake8: noqa: T201

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, get_args

from shapiq.explainer.tabular import TabularExplainerIndices
from shapiq.explainer.tree.treeshapiq import TreeSHAPIQIndices
from shapiq.game_theory.exact import ExactComputer
from shapiq.games.benchmark.treeshapiq_xai import TreeSHAPIQXAI
from tests.fixtures.data import get_california_housing_train_test_explain
from tests.fixtures.games import get_california_housing_imputer
from tests.fixtures.models import get_california_housing_random_forest

if TYPE_CHECKING:
    from shapiq.explainer.custom_types import ExplainerIndices
    from shapiq.games.base import Game
    from shapiq.interaction_values import InteractionValues


def _compute_values(
    game: Game,
    interaction_indices: list[ExplainerIndices],
    save_name: str,
    save_path: Path | None = None,
) -> dict[str, InteractionValues]:
    """Compute interaction values for the given game and save them to disk."""

    ivs = {}

    exact_computer = ExactComputer(game=game, n_players=game.n_players, evaluate_game=True)
    value_indices = ["SV", "BV"]
    for index in value_indices:
        iv = exact_computer(index=index, order=1)
        iv = iv.get_n_order(order=1)
        name = f"{save_name}_index={index}_order=1.pkl"
        if save_path is not None:
            iv.save(path=save_path / name)
        ivs[name] = iv
        print(f"Interaction values for index {index} (order 1):")
        print(iv)

    # compute Moebius as well
    iv = exact_computer(index="Moebius", order=game.n_players)
    name = f"{save_name}_index=Moebius_order={game.n_players}.pkl"
    if save_path is not None:
        iv.save(path=save_path / name)
    ivs[name] = iv
    print("Moebius interaction values:")
    print(iv)

    # compute interaction values for all indices that are in the ExplainerIndices
    orders = list(range(1, game.n_players + 1))
    for index in interaction_indices:
        if index in value_indices:
            continue
        for order in orders:
            iv = exact_computer(index=index, order=order)
            iv = iv.get_n_order(min_order=1, max_order=order)
            name = f"{save_name}_index={index}_order={order}.pkl"
            if save_path is not None:
                iv.save(path=save_path / name)
            ivs[name] = iv
            print(f"Interaction values for index {index} (order {order}):")
            print(iv)

    return ivs


def compute_tabular_explanations(save_path: Path | None = None) -> dict[str, InteractionValues]:
    """Compute explanations for the California Housing dataset using the Tabular Explainer."""
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
    assert imputer_hash == 9070456741283270540
    imputer.verbose = True
    return _compute_values(
        game=imputer,
        interaction_indices=list(get_args(TabularExplainerIndices)),
        save_name=f"iv_california_housing_imputer_{imputer_hash}",
        save_path=save_path,
    )


def compute_tree_explanations(save_path: Path | None = None) -> dict[str, InteractionValues]:
    """Compute explanations for the California Housing dataset using the TreeSHAPIQ Explainer."""
    x_train, y_train, x_test, y_test, x_explain = get_california_housing_train_test_explain()
    model = get_california_housing_random_forest()
    print(f"Model score: {model.score(x_test, y_test)}")
    print(f"Model prediction for x_explain: {model.predict(x_explain)}")

    game = TreeSHAPIQXAI(
        x=x_explain,
        tree_model=model,
        normalize=False,
        verbose=False,
    )

    # compute explanations
    return _compute_values(
        game=game,
        interaction_indices=list(get_args(TreeSHAPIQIndices)),
        save_name="iv_california_housing_tree",
        save_path=save_path,
    )


if __name__ == "__main__":
    SAVE_PATH = Path(__file__).parent.parent / "data" / "interaction_values" / "california_housing"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Creating and saving interaction values to {SAVE_PATH}")

    # compute the TreeSHAPIQ explanations for the California Housing dataset
    compute_tree_explanations(save_path=SAVE_PATH)

    # compute the tabular explanations for the California Housing dataset
    compute_tabular_explanations(save_path=SAVE_PATH)
