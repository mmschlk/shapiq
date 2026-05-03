"""PolySHAP — Shapley value approximation via polynomial regression.

Reference: Fumagalli et al. (2026), arXiv:2601.18608.
Repository: https://github.com/FFmgll/PolySHAP.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues


class PolySHAP(Approximator):
    """PolySHAP — Shapley value estimator via polynomial regression.

    Stub implementation: returns zero Shapley values for all players. Replace
    Step 2 in :meth:`approximate` with the interaction-informed polynomial
    regression algorithm.

    Args:
        n: The number of players.
        pairing_trick: If True, paired sampling is applied. Defaults to False.
        sampling_weights: Optional ``(n + 1,)`` array of coalition-size weights.
            If None, KernelSHAP weights are used. Defaults to None.
        random_state: Seed for deterministic sampling. Defaults to None.

    """

    valid_indices: tuple[str, ...] = ("SV",)

    def __init__(
        self,
        n: int,
        *,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n=n,
            max_order=1,
            index="SV",
            top_order=False,
            min_order=0,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )
        self.runtime_last_approximate_run: dict[str, float] = {}

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        """Approximate Shapley values via polynomial regression.

        TODO: replace the STUB block below with the real implementation.
        """
        # Step 1: empty-coalition baseline.
        empty_coalition = np.zeros((1, self.n), dtype=bool)
        baseline_value = float(game(empty_coalition)[0])

        # Step 2: STUB — replace before final submission.
        sv_values = np.zeros(self.n + 1, dtype=float)
        sv_values[0] = baseline_value

        # Step 3: canonical SV interaction_lookup.
        sv_lookup: dict[tuple[int, ...], int] = {(): 0}
        for i in range(self.n):
            sv_lookup[(i,)] = i + 1

        # Step 4: wrap and return.
        return InteractionValues(
            values=sv_values,
            index="SV",
            interaction_lookup=sv_lookup,
            baseline_value=baseline_value,
            min_order=0,
            max_order=1,
            n_players=self.n,
            estimated=(budget < 2 ** self.n),
            estimation_budget=int(budget),
        )
