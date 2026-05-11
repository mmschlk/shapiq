import numpy as np
from shapiq import InteractionValues

def interaction_values_to_dict(
    values: InteractionValues,
) -> dict[tuple[int, ...], float]:
    result = {}
    for interaction, pos in values.interaction_lookup.items():
        result[interaction] = float(values.values[pos])
    return result


def get_common_non_empty_interactions(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
) -> list[tuple[int, ...]]:
    gt_dict = interaction_values_to_dict(ground_truth)
    approx_dict = interaction_values_to_dict(approximation)

    return [
        interaction
        for interaction in gt_dict
        if interaction != () and interaction in approx_dict
    ]


def mse_metric(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
) -> float:
    gt_dict = interaction_values_to_dict(ground_truth)
    approx_dict = interaction_values_to_dict(approximation)
    common = get_common_non_empty_interactions(ground_truth, approximation)
    if not common:
        raise ValueError("No common interactions found.")
    gt_array = np.array([gt_dict[i] for i in common])
    approx_array = np.array([approx_dict[i] for i in common])
    return float(np.mean((gt_array - approx_array) ** 2))


def mae_metric(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
) -> float:
    gt_dict = interaction_values_to_dict(ground_truth)
    approx_dict = interaction_values_to_dict(approximation)
    common = get_common_non_empty_interactions(ground_truth, approximation)
    if not common:
        raise ValueError("No common interactions found.")
    gt_array = np.array([gt_dict[i] for i in common])
    approx_array = np.array([approx_dict[i] for i in common])
    return float(np.mean(np.abs(gt_array - approx_array)))