import numpy as np

from shapiq.approximator.regression.polyshap.polyshap import PolySHAP
from shapiq.utils.sets import powerset


class PolySHAPPartial(PolySHAP):
    """PolySHAP with a randomly-extended *partial* explanation frontier.

    The frontier always contains every singleton, then greedily adds
    higher-order interactions (in a randomly shuffled order) until
    ``n_explanation_terms`` terms have been selected.  The shuffling is
    driven by a random permutation of the player indices, making the frontier
    stochastic when ``random_state`` is not fixed.

    **When to use:** Practical fallback when the full *k*-additive frontier of
    :class:`PolySHAPKAdd` is too large for the available budget or memory (e.g.
    large *n* with order ≥ 2).  ``n_explanation_terms`` gives direct control
    over the regression complexity regardless of *n* or the interaction order.

    **Limitations:** The randomly chosen higher-order terms are not guaranteed
    to be the most informative ones, so accuracy can vary across runs when
    ``random_state`` is not fixed.  For the best systematic coverage at a given
    order, prefer :class:`PolySHAPKAdd`; if domain knowledge is available,
    prefer :class:`PolySHAPPrior`.

    Args:
        n: The number of players.
        n_explanation_terms: Total number of frontier terms (including
            singletons and the empty coalition).
        sizes_to_exclude: Coalition sizes to skip when extending the
            frontier beyond singletons. Defaults to ``None``.
        pairing_trick: If ``True``, the pairing trick is applied. Defaults to ``False``.
        sampling_weights: Optional sampling weights of shape ``(n + 1,)``.
        random_state: Random state used to shuffle interaction candidates
            and, via the parent, for coalition sampling. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int,
        n_explanation_terms: int,
        sizes_to_exclude: set[int] | None = None,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        explanation_frontier: dict[tuple, int] = {}
        pos = 0

        # Always include all singletons first.
        for S in powerset(range(n), max_size=1):
            explanation_frontier[S] = pos
            pos += 1

        # Extend with higher-order interactions in a reproducible random order.
        if random_state is not None:
            np.random.seed(random_state)
        perm = list(range(n))
        np.random.shuffle(perm)

        for S in powerset(range(n), min_size=2):
            if sizes_to_exclude is not None and len(S) in sizes_to_exclude:
                continue
            if pos >= n_explanation_terms:
                break
            explanation_frontier[tuple(sorted(perm[i] for i in S))] = pos
            pos += 1

        super().__init__(
            n=n,
            explanation_frontier=explanation_frontier,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )