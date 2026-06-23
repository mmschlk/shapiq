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
        # Defensive check: fetch tree model from game directly or via game.setup
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

        # 3. Dynamic Min-Order Alignment
        # Standard benchmark interaction evaluation runs from order 1 up to max_order.
        # TreeExplainer defaults to min_order=0; we force it to 1 and remove the empty set ().
        gt_values.min_order = 1
        if () in gt_values.interactions:
            del gt_values.interactions[()]

        # 4. Universal Key Padding (Zero-Player Axiom Alignment)
        # Pad any missing subsets (due to un-split tree features) with 0.0 to match the
        # full player set dimension expected by black-box approximators.
        expected_keys = powerset(
            range(game.n_players), min_size=gt_values.min_order, max_size=max_order
        )
        for subset in expected_keys:
            if subset not in gt_values.interactions:
                gt_values.interactions[subset] = 0.0

        return gt_values
    # Fallback to the black-box exact brute-force computer for any other method or if TreeExplainer is not applicable.
    exact = ExactComputer(game=game)
    return exact(index=index, order=max_order)
