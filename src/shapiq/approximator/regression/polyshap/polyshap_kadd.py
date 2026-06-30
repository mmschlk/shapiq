"""k-additive PolySHAP approximator (:class:`PolySHAPKAdd`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.approximator.regression.polyshap.polyshap import PolySHAP
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    import numpy as np


class PolySHAPKAdd(PolySHAP):
    """PolySHAP with a *k*-additive explanation frontier.

    The frontier consists of all subsets of players up to size ``max_order``,
    optionally skipping certain coalition sizes.  The frontier size equals
    ``sum(C(n, k) for k in range(max_order + 1))``, which grows rapidly with
    *n* and *max_order*.

    **When to use:** Default choice when a systematic, exhaustive treatment of
    all interactions up to order *k* is feasible.  Produces a deterministic,
    reproducible frontier and yields the best accuracy in practice when the
    budget comfortably exceeds the frontier size.  At ``max_order=1`` it
    reduces exactly to KernelSHAP.

    **Limitations:** Frontier size can become prohibitively large for high *n*
    or high *max_order* (e.g. order-2 with n=20 already adds 190 pairwise
    terms).  Prefer :class:`PolySHAPPartial` when the budget or memory is
    tight, or :class:`PolySHAPPrior` when domain knowledge identifies the
    relevant interactions.

    Args:
        n: The number of players.
        max_order: Maximum coalition size to include in the frontier.
        sizes_to_exclude: Coalition sizes to omit from the frontier.
            Defaults to ``None`` (no sizes excluded).
        pairing_trick: If ``True``, the pairing trick is applied. Defaults to ``False``.
        sampling_weights: Optional sampling weights of shape ``(n + 1,)``.
        random_state: Random state for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        *,
        sizes_to_exclude: set[int] | None = None,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the k-additive PolySHAP approximator."""
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
