# Benchmark Metrics and Scoring

The benchmark separates three responsibilities:

* A metric computes one value from `ground_truth` and `estimated` values.
* The `Scorer` computes the complete `run_record["metrics"]` dictionary for one run.
* The Aggregator averages existing metric values across successful run records and seeds.

The intended flow is:

```text
Runner
  -> obtains ground_truth and estimated interaction values
  -> Scorer computes run_record["metrics"]
  -> run_record is stored
  -> Aggregator averages metrics across run_records/seeds
  -> Leaderboard/UI consumes aggregated records
```

The Aggregator does not recompute metrics and the Scorer does not aggregate across runs.

## Metric Registry

Metrics are registered in `src/leaderboard/metrics/registry.py`.
Each entry has a `MetricSpec` with:

* `name`: the canonical run-record key.
* `function`: the metric implementation.
* `higher_is_better`: whether larger values should rank better.
* `category`: a coarse group such as `error`, `faithfulness`, `rank_correlation`, or `top_k`.
* `description`: a short human-readable explanation.

The canonical metric keys are the keys stored in `run_record["metrics"]` and consumed by aggregation.
For example, normalized MSE is stored as `mse_normalized` to match the existing run-record format.
The older name `normalized_mse` is accepted as a Scorer input alias but is not emitted as a separate stored key.

## Available Metrics

### Mean Squared Error (`mse`)

Mean squared error measures the average squared difference between the reference and estimated values:

```text
mse = mean((ground_truth - estimated)^2)
```

Lower values are better.
A value of `0.0` means the estimated values match the reference values exactly.

### Mean Absolute Error (`mae`)

Mean absolute error measures the average absolute difference:

```text
mae = mean(abs(ground_truth - estimated))
```

Lower values are better.
Compared with MSE, MAE weights all absolute deviations linearly.

### Normalized Mean Squared Error (`mse_normalized`)

Normalized mean squared error divides the mean squared error by the variance of the reference values:

```text
mse_normalized = mse / variance(ground_truth)
```

Lower values are better.
This makes the error easier to compare across benchmark instances with different value scales.
If the variance of `ground_truth` is `0`, the implementation returns the unnormalized MSE because there is no reference variance to normalize by.

### R2 Faithfulness (`r2`)

R2 is a reconstruction or faithfulness score, not a distance metric:

```text
r2 = 1 - sum((estimated - ground_truth)^2) / sum((ground_truth - mean(ground_truth))^2)
```

Higher values are better.
A value of `1.0` means perfect reconstruction, `0.0` means equal to predicting the reference mean, and negative values mean worse than predicting the reference mean.
If the denominator is zero because the reference values are constant, the implementation returns `NaN`.

This follows the ProxySPEX faithfulness definition from Section 3.1, Equation (2): Butler, Agarwal, Kang, Erginbas, Yu, and Ramchandran, "ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs," NeurIPS 2025.

### Spearman Rank Correlation (`spearman`)

Spearman correlation evaluates whether the estimated values preserve the ranking of the reference values.
It is useful when the relative order of important interactions matters more than matching their absolute magnitudes.

Higher values are better.
The score ranges from `-1.0` to `1.0` in regular cases.
If SciPy returns `NaN`, for example for constant inputs where the correlation is undefined, the implementation reports `0.0`.

### Kendall Tau (`kendall_tau`)

Kendall tau is another rank-correlation metric.
It compares pairwise ordering agreement between reference and estimated values.

Higher values are better.
As with Spearman, undefined SciPy results are reported as `0.0`.

### Precision at K (`precision_at_k`)

Precision at K measures overlap between the top-`k` absolute reference values and the top-`k` absolute estimated values:

```text
precision_at_k = |top_k(abs(ground_truth)) intersect top_k(abs(estimated))| / k
```

Higher values are better.
The default `k` is `10`, and callers can override it through `metric_params`.

## Scoring One Run

Use `Scorer` when a benchmark run already has aligned `ground_truth` and `estimated` values and needs the dictionary stored in `run_record["metrics"]`:

```python
import numpy as np

from metrics.scorer import Scorer

scorer = Scorer(
    metric_names=["mse", "mse_normalized", "r2", "spearman", "precision_at_k"],
    metric_params={"precision_at_k": {"k": 10}},
)

ground_truth = np.array([1.0, 2.0, 3.0])
estimated = np.array([1.1, 1.9, 3.2])

metrics = scorer.score(ground_truth, estimated)
```

The returned object is a plain `dict[str, float | None]` using canonical run-record keys.
Unselected metrics are included with `None` so the output remains compatible with the Aggregator.
By default, a metric failure produces `None` for that metric.
Use `Scorer(..., fail_fast=True)` to re-raise metric exceptions instead.

`compute_all_metrics(ground_truth, estimated)` remains available as a compatibility wrapper around `Scorer().score(...)`.

## Adding a Metric

To add another metric, implement the `Metric` interface and register it in `METRIC_SPECS`:

```python
from .base import Metric
from .result import MetricResult


class ExampleMetric(Metric):
    name = "example"
    higher_is_better = True

    def compute(self, ground_truth, estimated, **params) -> MetricResult:
        value = 0.0
        return MetricResult(
            metric_name=self.name,
            value=value,
            higher_is_better=self.higher_is_better,
        )
```

Registering the metric adds it to `METRIC_KEYS`, makes it available to `Scorer`, and lets the Aggregator include the key without duplicating metric definitions.
