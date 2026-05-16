from .registry import METRICS


def compute_all_metrics(ground_truth, estimated):

    results = {}

    for name, metric in METRICS.items():

        metric_result = metric.compute(
            ground_truth,
            estimated,
        )

        results[name] = metric_result.value

    return results