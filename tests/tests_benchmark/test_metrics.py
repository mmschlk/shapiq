"""This test module contains all tests for computing error metrics."""

import random

import numpy as np
import pytest

from shapiq import InteractionValues, powerset
from shapiq.benchmark.metrics import get_all_metrics


@pytest.mark.parametrize(
    "index_gt, index_et, order_gt, order_et, warning_expected",
    [
        ("SV", "SV", 1, 1, False),
        ("SII", "SII", 2, 2, False),
        ("SV", "kADD-SHAP", 1, 1, False),  # no Warning (W) because kADD-SHAP is converted to SV
        ("SV", "FSII", 1, 2, True),  # W: FSII order 2 is not SV of order 1
        ("FSII", "SII", 2, 2, True),  # W: FSII is not SII
        ("SV", "kADD-SHAP", 1, 2, False),  # no W: kADD-SHAP order >1 is turned into SV and matches
    ],
)
def test_computation(index_gt, index_et, order_gt, order_et, warning_expected):

    n_players = 5

    gt = [random.random() for _ in powerset(range(n_players), min_size=0, max_size=order_gt)]
    gt = InteractionValues(
        values=np.array(gt),
        index=index_gt,
        max_order=order_gt,
        min_order=0,
        n_players=n_players,
        baseline_value=0.0,
    )
    et = [random.random() for _ in powerset(range(n_players), min_size=0, max_size=order_et)]
    et = InteractionValues(
        values=np.array(et),
        index=index_et,
        max_order=order_et,
        min_order=order_et,
        n_players=n_players,
        baseline_value=0.0,
    )

    if warning_expected:
        with pytest.warns(UserWarning):
            _ = get_all_metrics(ground_truth=gt, estimated=et)
    else:
        _ = get_all_metrics(ground_truth=gt, estimated=et)

    # run with order indicator
    _ = get_all_metrics(ground_truth=gt, estimated=et, order_indicator="1")
