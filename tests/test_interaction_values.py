from __future__ import annotations

import numpy as np

from shapiq.interaction_values import InteractionValues


def test_interaction_values_eq():
    iv1 = InteractionValues(
        index="SV",
        max_order=1,
        min_order=0,
        estimated=False,
        estimation_budget=None,
        n_players=8,
        baseline_value=2.0719469373788755,
        values=np.array([0.0] * 300),
    )
    iv2 = InteractionValues(
        index="SV",
        max_order=1,
        min_order=0,
        estimated=False,
        estimation_budget=None,
        n_players=8,
        baseline_value=2.071946937378876,  # Note the slight difference
        values=np.array([0.0] * 300),
    )
    assert iv1 == iv2
