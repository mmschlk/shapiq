from shapiq import InteractionValues
from custom_types import MetricFunction

def compute_metrics(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
    metrics: dict[str, MetricFunction],
) -> dict[str, float]:
    metric_results: dict[str, float] = {}

    for metric_name, metric_func in metrics.items():
        metric_results[metric_name] = metric_func(
            ground_truth,
            approximation,
        )

    return metric_results