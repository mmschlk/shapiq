from .registry import METRICS
from .utils import remove_empty_value_if_needed


def compute_all_metrics(ground_truth, estimated):
    ground_truth = remove_empty_value_if_needed(ground_truth)
    estimated = remove_empty_value_if_needed(estimated)

    results = {}

    for name, metric in METRICS.items():

        metric_result = metric.compute(ground_truth,estimated)
        results[name] = metric_result.value

    return results