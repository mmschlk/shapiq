"""Metrics for evaluating the performance of interaction values."""

import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import count_interactions, powerset


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
    except KeyError:
        return interaction

    return interaction


def compute_mse(ground_truth: InteractionValues, estimated: InteractionValues) -> float:
    """Compute the mean squared error between two interaction values."""
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    return float(np.mean(diff_values**2))


def compute_mse_old(ground_truth: InteractionValues, estimated: InteractionValues) -> float:
    """Compute the mean squared error the old way."""
    n_players, order = ground_truth.n_players, ground_truth.max_order
    assert ground_truth.max_order == ground_truth.min_order, "Only works for single order."
    n_interactions = count_interactions(n_players, min_order=order, max_order=order)
    gt_arr = [
        ground_truth[interaction]
        for interaction in powerset(range(ground_truth.n_players), min_size=order, max_size=order)
    ]
    gt_arr = np.array(gt_arr)
    et_arr = [
        estimated[interaction]
        for interaction in powerset(range(estimated.n_players), min_size=order, max_size=order)
    ]
    et_arr = np.array(et_arr)
    return np.sum((gt_arr - et_arr) ** 2) / n_interactions


def compute_mae(ground_truth: InteractionValues, estimated: InteractionValues) -> float:
    """Compute the mean absolute error between two interaction values."""
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    return float(np.mean(np.abs(diff_values)))


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
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = 10
) -> dict:
    """Get all metrics for the interaction values."""
    metrics = {
        "MSE": compute_mse(ground_truth, estimated),
        "MSE_old": compute_mse_old(ground_truth, estimated),
        "MAE": compute_mae(ground_truth, estimated),
        "Precision@k": compute_precision_at_k(ground_truth, estimated, k=k),
        "Precision@k_k": k,
    }
    return metrics
