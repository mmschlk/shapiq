from shapiq.approximator.regression.polyshap.polyshap import PolySHAP
from shapiq.utils.sets import powerset

import numpy as np


class PolySHAPKAdd(PolySHAP):
    """PolySHAP with a *k*-additive explanation frontier.

    The frontier consists of all subsets of players up to size ``max_order``,
    optionally skipping certain coalition sizes.

    Args:
        n: The number of players.
        max_order: Maximum coalition size to include in the frontier.
        sizes_to_exclude: Coalition sizes to omit from the frontier.
            Defaults to ``None`` (no sizes excluded).
        pairing_trick: If ``True``, the pairing trick is applied. Defaults to ``False``.
        sampling_weights: Optional sampling weights of shape ``(n + 1,)``.
        replacement: Whether to sample with replacement. Defaults to ``True``.
        random_state: Random state for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        sizes_to_exclude: set[int] | None = None,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        explanation_frontier: dict[tuple, int] = {}
        pos = 0
        for S in powerset(range(n), max_size=max_order):
            if sizes_to_exclude is None or len(S) not in sizes_to_exclude:
                explanation_frontier[S] = pos
                pos += 1

        super().__init__(
            n=n,
            explanation_frontier=explanation_frontier,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )