"""Integration test for the Product Kernel Explainer using the California Housing dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, get_args

from shapiq.explainer.product_kernel.conversion import convert_svm
from shapiq.explainer.product_kernel.game import ProductKernelGame
from shapiq.explainer.product_kernel.product_kernel import ProductKernelSHAPIQIndices
from shapiq.game_theory.exact import ExactComputer
from tests.shapiq.fixtures.data import get_california_housing_train_test_explain
from tests.shapiq.fixtures.models import get_california_housing_svr

if TYPE_CHECKING:
    from shapiq.explainer.custom_types import ValidProductKernelExplainerIndices
    from shapiq.game import Game
    from shapiq.interaction_values import InteractionValues


def _compute_values(
    game: Game,
    interaction_indices: list[ValidProductKernelExplainerIndices],
    save_name: str,
    save_path: Path | None = None,
) -> dict[str, InteractionValues]:
    """Compute interaction values for the given game and save them to disk."""

    ivs = {}

    exact_computer = ExactComputer(game=game, n_players=game.n_players, evaluate_game=True)
    value_indices = ["SV"]
    for index in value_indices:
        iv = exact_computer(index=index, order=1)
        iv = iv.get_n_order(order=1)
        name = f"{save_name}_index={index}_order=1.json"
        if save_path is not None:
            iv.save(path=save_path / name)
        ivs[name] = iv
        print(iv)  # noqa: T201

    # compute interaction values for all indices that are in the ValidProductKernelExplainerIndices
    orders = list(range(1, game.n_players + 1))
    for index in interaction_indices:
        if index in value_indices:
            continue
        for order in orders:
            iv = exact_computer(index=index, order=order)
            iv = iv.get_n_order(min_order=1, max_order=order)
            name = f"{save_name}_index={index}_order={order}.json"
            if save_path is not None:
                iv.save(path=save_path / name)
            ivs[name] = iv
            print(iv)  # noqa: T201

    return ivs


def compute_product_kernel_explanations(
    save_path: Path | None = None,
) -> dict[str, InteractionValues]:
    """Compute explanations for the California Housing dataset using the ProductKernel Explainer."""
    x_train, y_train, x_test, y_test, x_explain = get_california_housing_train_test_explain()
    model = get_california_housing_svr()
    converted_model = convert_svm(model)
    print(f"Model score: {model.score(x_test, y_test)}")  # noqa: T201
    print(f"Model prediction for x_explain: {model.predict(x_explain)}")  # noqa: T201

    game = ProductKernelGame(
        model=converted_model,
        n_players=x_explain.shape[1],
        explain_point=x_explain[0],
        normalize=True,
    )

    # compute explanations
    return _compute_values(
        game=game,
        interaction_indices=list(get_args(ProductKernelSHAPIQIndices)),
        save_name="iv_california_housing_pk",
        save_path=save_path,
    )


if __name__ == "__main__":
    SAVE_PATH = Path(__file__).parent.parent / "data" / "interaction_values" / "california_housing"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Creating and saving interaction values to {SAVE_PATH}")  # noqa: T201

    # compute the TreeSHAPIQ explanations for the California Housing dataset
    compute_product_kernel_explanations(save_path=SAVE_PATH)
