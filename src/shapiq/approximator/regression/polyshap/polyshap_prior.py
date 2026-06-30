"""User-prior PolySHAP approximator (:class:`PolySHAPPrior`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.approximator.regression.polyshap.polyshap import PolySHAP

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np


class PolySHAPPrior(PolySHAP):
    """PolySHAP with a user-supplied *prior* explanation frontier.

    The frontier is taken directly from ``q_prior``, which should be an
    iterable of coalition tuples ordered as desired.  The caller is
    responsible for ensuring all singletons ``(i,)`` for ``i in range(n)``
    are present; the parent class will raise ``ValueError`` otherwise.

    **When to use:** Best when domain knowledge identifies a specific set of
    interactions likely to explain most of the model's behaviour.  A
    well-informed prior can match or outperform :class:`PolySHAPKAdd` at the
    same frontier size by focusing capacity on interactions that actually
    matter.  Also useful for reproducing the exact frontier from a prior
    experiment.

    **Limitations:** Accuracy degrades badly if the prior is uninformative or
    wrong — more so than the systematic alternatives.  Requires the caller to
    construct and validate the frontier manually.  When no domain knowledge is
    available, prefer :class:`PolySHAPKAdd` (systematic) or
    :class:`PolySHAPPartial` (budget-controlled random extension).

    Args:
        n: The number of players.
        q_prior: An iterable of coalition tuples that defines the frontier
            and its column ordering.  Must include all singletons.
        pairing_trick: If ``True``, the pairing trick is applied. Defaults to ``False``.
        sampling_weights: Optional sampling weights of shape ``(n + 1,)``.
        random_state: Random state for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int,
        q_prior: Iterable[tuple[int, ...]],
        *,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the user-prior PolySHAP approximator."""
        explanation_frontier: dict[tuple, int] = {S: pos for pos, S in enumerate(q_prior)}

        super().__init__(
            n=n,
            explanation_frontier=explanation_frontier,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )
