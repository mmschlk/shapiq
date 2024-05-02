"""Metrics for evaluating the performance of interaction values."""

from typing import Optional

import numpy as np
from scipy.stats import kendalltau

from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset


def _remove_empty_value(interaction: InteractionValues) -> InteractionValues:
    """Manually sets the empty value to zero.

    Args:
        interaction: The interaction values to remove the empty value from.

    Returns:
        The interaction values without the empty value.
    """
    try:
        empty_index = interaction.interaction_lookup[()]
        interaction.values[empty_index] = 0
        return interaction
    except KeyError:
        return interaction


def compute_mse(ground_truth: InteractionValues, estimated: InteractionValues) -> float:
    """Compute the mean squared error between two interaction values."""
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    return float(np.mean(diff_values**2))


def compute_mae(ground_truth: InteractionValues, estimated: InteractionValues) -> float:
    """Compute the mean absolute error between two interaction values."""
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    return float(np.mean(np.abs(diff_values)))


def compute_kendall_tau(ground_truth: InteractionValues, estimated: InteractionValues) -> float:
    """Compute the Kendall Tau between two interaction values."""
    # get the interactions as a sorted array
    gt_values, estimated_values = [], []
    for interaction in powerset(
        range(ground_truth.n_players),
        min_size=ground_truth.min_order,
        max_size=ground_truth.max_order,
    ):
        gt_values.append(ground_truth[interaction])
        estimated_values.append(estimated[interaction])
    # array conversion
    gt_values, estimated_values = np.array(gt_values), np.array(estimated_values)
    # sort the values
    gt_indices, estimated_indices = np.argsort(gt_values), np.argsort(estimated_values)
    # compute the Kendall Tau
    tau, _ = kendalltau(gt_indices, estimated_indices)
    return tau


def compute_precision_at_k(
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = 10
) -> float:
    """Compute the precision at k between two interaction values."""
    ground_truth_values = _remove_empty_value(ground_truth)
    estimated_values = _remove_empty_value(estimated)
    top_k, _ = ground_truth_values.get_top_k_interactions(k=k)
    top_k_estimated, _ = estimated_values.get_top_k_interactions(k=k)
    precision_at_k = len(set(top_k.keys()).intersection(set(top_k_estimated.keys()))) / k
    return precision_at_k


def get_all_metrics(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
    order_indicator: Optional[str] = None,
) -> dict:
    """Get all metrics for the interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        order_indicator: The order indicator for the metrics. Defaults to None.

    Returns:
        The metrics as a dictionary.
    """
    if order_indicator is None:
        order_indicator = ""
    else:
        order_indicator += "_"

    metrics = {
        order_indicator + "MSE": compute_mse(ground_truth, estimated),
        order_indicator + "MAE": compute_mae(ground_truth, estimated),
        order_indicator + "Precision@10": compute_precision_at_k(ground_truth, estimated, k=10),
        order_indicator + "Precision@5": compute_precision_at_k(ground_truth, estimated, k=5),
        order_indicator + "KendallTau": compute_kendall_tau(ground_truth, estimated),
    }
    return metrics
