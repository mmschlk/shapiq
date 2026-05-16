import numpy as np

from .base import Metric
from .result import MetricResult
from .utils import remove_empty_value

class MSEMetric(Metric):
    name = "mse"
    higher_is_better = False

    def compute(self, ground_truth, estimated) -> MetricResult:
        difference = ground_truth - estimated

        return MetricResult(
            metric_name=self.name,
            value=float(np.mean(difference ** 2)),
            higher_is_better=self.higher_is_better,
        )


class NormalizedMSEMetric(Metric):
    name = "normalized_mse"
    higher_is_better = False

    def compute(self, ground_truth, estimated) -> MetricResult:
        difference = ground_truth - estimated
        mse = np.mean(difference ** 2)
        variance = np.var(ground_truth)

        if variance == 0:
            value = float(mse)
        else:
            value = float(mse / variance)

        return MetricResult(
            metric_name=self.name,
            value=value,
            higher_is_better=self.higher_is_better,
        )