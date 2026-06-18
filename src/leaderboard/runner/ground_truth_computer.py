"""Ground truth computer for the leaderboard runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq import ExactComputer
from shapiq.tree.explainer import TreeExplainer
from shapiq.utils import powerset

if TYPE_CHECKING:
    from leaderboard.runner.custom_types import InteractionIndex
    from shapiq import InteractionValues
    from shapiq.game import Game


def compute_ground_truth(
    game: Game,
    index: InteractionIndex,
    max_order: int,
    method: str = "ExactComputer",
) -> InteractionValues:
    """Compute exact interaction values for a game.

    Args:
        game: The game for which exact interaction values are computed.
        index: The interaction index to compute.
        max_order: The maximum interaction order to compute.
        method: The method to use ("ExactComputer" or "TreeExplainer").

    Returns:
        The exact interaction values.
    """
    if method == "TreeExplainer":
        # Defensive check to fetch the wrapped tree model from the game setup
        model = getattr(game, "model", None)
        if model is None and hasattr(game, "setup"):
            model = getattr(game.setup, "model", None)

        if model is None:
            raise AttributeError(
                f"The current game '{game.__class__.__name__}' cannot provide a valid tree model for TreeExplainer!"
            )

        # Initialize and run TreeExplainer on the target instance
        explainer = TreeExplainer(model=model, index=index, max_order=max_order)
        gt_values = explainer.explain(game.x)

        # Sanitize and align TreeExplainer properties with the expected benchmark format
        gt_values.index = index

        # If the game is normalized, align the baseline value to 0.0
        if getattr(game, "normalize", False):
            gt_values.baseline_value = 0.0

        # For standard SV evaluation, align min_order and remove the baseline empty set ()
        if index == "SV":
            gt_values.min_order = 1
            if () in gt_values.interactions:
                del gt_values.interactions[()]

        # 💡 Pad missing interaction keys with 0.0 to match the full player set of the game.
        # This resolves the mismatch caused by features that were never split on in the tree model.
        for subset in powerset(
            range(game.n_players), min_size=gt_values.min_order, max_size=max_order
        ):
            if subset not in gt_values.interactions:
                gt_values.interactions[subset] = 0.0

        return gt_values
    # Fallback to the black-box exact brute-force computer for any other method or if TreeExplainer is not applicable.
    exact = ExactComputer(game=game)
    return exact(index=index, order=max_order)
