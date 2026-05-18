from scipy.stats import spearmanr

from .base import Metric
from .result import MetricResult

class SpearmanMetric(Metric):
        name = "spearman"
        higher_is_better = True

        def compute(self, ground_truth, estimated) -> MetricResult:
            correlation, _ = spearmanr(ground_truth, estimated)

            if correlation != correlation:  # NaN
                correlation = 0.0

            return MetricResult(
                metric_name=self.name,
                value=float(correlation),
                higher_is_better=self.higher_is_better,
            )