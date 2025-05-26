"""Metrics for evaluating the performance of interaction values."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau

from shapiq.utils.sets import count_interactions, powerset

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

__all__ = ["get_all_metrics"]


def _remove_empty_value(interaction: InteractionValues) -> InteractionValues:
    """Manually sets the empty value (e.g. baseline value) to zero in the values array.

    Args:
        interaction: The interaction values to remove the empty value from.

    Returns:
        The interaction values without the empty value.

    """
    try:
        _ = interaction.interaction_lookup[()]
        new_interaction = copy.deepcopy(interaction)
        empty_index = new_interaction.interaction_lookup[()]
        new_interaction.values[empty_index] = 0
    except KeyError:
        return interaction
    return new_interaction


def compute_diff_metrics(ground_truth: InteractionValues, estimated: InteractionValues) -> dict:
    """Compute metrics via the difference between the ground truth and estimated interaction values.

    Computes the following metrics:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - Sum of Squared Errors (SSE)
        - Sum of Absolute Errors (SAE)

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.

    Returns:
        The metrics between the ground truth and estimated interaction values.

    """
    try:
        difference = ground_truth - estimated
    except ValueError:  # maybe indices don't want to match
        if ground_truth.index != estimated.index:
            if {ground_truth.index, estimated.index} == {"SV", "kADD-SHAP"}:
                sv_values = ground_truth if ground_truth.index == "SV" else estimated
                kadd_values = ground_truth if ground_truth.index == "kADD-SHAP" else estimated
                kadd_values = kadd_values.get_n_order(order=1)  # make kADD-SHAP same order as SV
                difference = sv_values - kadd_values
            else:
                if ground_truth.index == "SV":
                    estimated_values = estimated.get_n_order(order=1, min_order=0)
                    estimated_values.index = "SV"
                    ground_truth_values = ground_truth
                else:
                    estimated_values = estimated
                    ground_truth_values = copy.deepcopy(ground_truth)
                    ground_truth_values.index = estimated.index
                warnings.warn(
                    f"Indices do not match for {ground_truth.index} and {estimated.index}. Will "
                    f"compare anyway but results need to be interpreted with care.",
                    stacklevel=2,
                )
                difference = ground_truth_values - estimated_values
        else:
            raise
    diff_values = _remove_empty_value(difference).values
    n_values = count_interactions(
        ground_truth.n_players,
        ground_truth.max_order,
        ground_truth.min_order,
    )
    return {
        "MSE": np.sum(diff_values**2) / n_values,
        "MAE": np.sum(np.abs(diff_values)) / n_values,
        "SSE": np.sum(diff_values**2),
        "SAE": np.sum(np.abs(diff_values)),
    }


def compute_kendall_tau(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
    k: int | None = None,
) -> float:
    """Compute the Kendall Tau between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top-k ground truth values to consider. Defaults to `None`, which considers all
            interactions.

    Returns:
        The Kendall Tau between the ground truth and estimated interaction values.

    """
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
    if k is not None:
        gt_indices, estimated_indices = gt_indices[:k], estimated_indices[:k]
    # compute the Kendall Tau
    tau, _ = kendalltau(gt_indices, estimated_indices)
    return tau


def compute_precision_at_k(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
    k: int = 10,
) -> float:
    """Compute the precision at k between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top-k ground truth values to consider. Defaults to 10.

    Returns:
        The precision at k between the ground truth and estimated interaction values.

    """
    ground_truth_values = _remove_empty_value(ground_truth)
    estimated_values = _remove_empty_value(estimated)
    top_k, _ = ground_truth_values.get_top_k(k=k, as_interaction_values=False)
    top_k_estimated, _ = estimated_values.get_top_k(k=k, as_interaction_values=False)
    return len(set(top_k.keys()).intersection(set(top_k_estimated.keys()))) / k


def get_all_metrics(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
    order_indicator: str | None = None,
) -> dict:
    """Get all metrics for the interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        order_indicator: An optional order indicator to prepend to the metrics. Defaults to `None`.

    Returns:
        The metrics as a dictionary.

    """
    if order_indicator is None:
        order_indicator = ""
    else:
        order_indicator += "_"

    metrics = {
        order_indicator + "Precision@10": compute_precision_at_k(ground_truth, estimated, k=10),
        order_indicator + "Precision@5": compute_precision_at_k(ground_truth, estimated, k=5),
        order_indicator + "KendallTau": compute_kendall_tau(ground_truth, estimated),
        order_indicator + "KendallTau@10": compute_kendall_tau(ground_truth, estimated, k=10),
        order_indicator + "KendallTau@50": compute_kendall_tau(ground_truth, estimated, k=50),
    }

    # get diff metrics
    metrics_diff = compute_diff_metrics(ground_truth, estimated)
    if order_indicator != "":  # add the order indicator to the diff metrics
        metrics_diff = {order_indicator + key: value for key, value in metrics_diff.items()}

    metrics.update(metrics_diff)
    return metrics
