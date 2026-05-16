# Benchmark Metrics

The living benchmark compares estimated interaction values against a reference result with a small set of metrics.
Each metric receives two array-like inputs:

* `ground_truth`: the reference interaction values.
* `estimated`: the values produced by an approximation method.

Metrics are available through the `METRICS` registry in `src/metrics/registry.py`.
The registry maps the public metric name to a metric object with a `compute(ground_truth, estimated)` method.
Each computation returns a `MetricResult` containing the metric name, the numeric value, and whether larger values are better.

## Available Metrics

### Mean Squared Error (`mse`)

Mean squared error measures the average squared difference between the reference and estimated values:

```text
mse = mean((ground_truth - estimated)^2)
```

Lower values are better.
A value of `0.0` means the estimated values match the reference values exactly.
Because errors are squared, larger deviations have a stronger effect on the score than smaller deviations.

### Normalized Mean Squared Error (`normalized_mse`)

Normalized mean squared error divides the mean squared error by the variance of the reference values:

```text
normalized_mse = mse / variance(ground_truth)
```

Lower values are better.
This makes the error easier to compare across benchmark instances with different value scales.
If the variance of `ground_truth` is `0`, the implementation returns the unnormalized MSE because there is no reference variance to normalize by.

### Spearman Rank Correlation (`spearman`)

Spearman correlation evaluates whether the estimated values preserve the ranking of the reference values.
It is useful when the relative order of important interactions matters more than matching their absolute magnitudes.

Higher values are better.
The score ranges from `-1.0` to `1.0` in regular cases:

* `1.0` means the rankings agree perfectly.
* `0.0` means there is no rank correlation.
* `-1.0` means the rankings are exactly reversed.

If SciPy returns `NaN`, for example for constant inputs where the correlation is undefined, the implementation reports `0.0`.

## Computing All Metrics

Use `compute_all_metrics` to evaluate every registered metric at once:

```python
import numpy as np

from metrics.evaluator import compute_all_metrics

ground_truth = np.array([1.0, 2.0, 3.0])
estimated = np.array([1.1, 1.9, 3.2])

scores = compute_all_metrics(ground_truth, estimated)
```

The returned dictionary maps metric names to numeric values:

```python
{
    "mse": 0.020000000000000035,
    "normalized_mse": 0.030000000000000054,
    "spearman": 1.0,
}
```

Use the `higher_is_better` flag from the corresponding `MetricResult` or metric object when sorting leaderboard results.
For `mse` and `normalized_mse`, smaller scores are better.
For `spearman`, larger scores are better.

## Adding a Metric

To add another metric, implement the `Metric` interface and register an instance in `METRICS`:

```python
from .base import Metric
from .result import MetricResult


class ExampleMetric(Metric):
    name = "example"
    higher_is_better = True

    def compute(self, ground_truth, estimated) -> MetricResult:
        value = 0.0
        return MetricResult(
            metric_name=self.name,
            value=value,
            higher_is_better=self.higher_is_better,
        )
```

Then add it to `src/metrics/registry.py` so `compute_all_metrics` includes it automatically.
