from shapiq.approximator.regression.polyshap.polyshap import PolySHAP

import numpy as np


class PolySHAPPrior(PolySHAP):
    """PolySHAP with a user-supplied *prior* explanation frontier.

    The frontier is taken directly from ``Q_prior``, which should be an
    iterable of coalition tuples ordered as desired.  The caller is
    responsible for ensuring all singletons are present.

    Args:
        n: The number of players.
        q_prior: An iterable of coalition tuples that defines the frontier
            and its column ordering.
        pairing_trick: If ``True``, the pairing trick is applied. Defaults to ``False``.
        sampling_weights: Optional sampling weights of shape ``(n + 1,)``.
        random_state: Random state for reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int,
        q_prior,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        explanation_frontier: dict[tuple, int] = {S: pos for pos, S in enumerate(q_prior)}

        super().__init__(
            n=n,
            explanation_frontier=explanation_frontier,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )