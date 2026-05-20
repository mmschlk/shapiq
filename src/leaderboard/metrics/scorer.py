"""Scoring utilities for benchmark metrics."""

from __future__ import annotations

from .registry import METRIC_ALIASES, METRIC_KEYS, METRIC_SPECS
from .utils import prepare_metric_inputs


class Scorer:
    """Compute the complete metrics dictionary for one benchmark run.

    The runner uses this class after it has already computed ground truth and
    approximated values. The scorer does not run approximators and does not
    aggregate across seeds; it only evaluates the selected metrics for one pair
    of aligned inputs.
    """

    def __init__(
        self,
        metric_names: list[str] | None = None,
        metric_params: dict[str, dict[str, object]] | None = None,
        fail_fast: bool = False,  # noqa: FBT001, FBT002 - Preserve public constructor API.
    ) -> None:
        """Create a scorer for selected metrics.

        Args:
            metric_names: Metric names to compute. ``None`` means all canonical
                metric keys from the registry. Aliases such as
                ``normalized_mse`` are normalized to canonical run-record keys.
            metric_params: Optional per-metric keyword arguments, for example
                ``{"precision_at_k": {"k": 10}}``.
            fail_fast: If ``True``, re-raise metric errors. If ``False``, store
                ``None`` for a metric that fails and continue scoring.
        """
        self.metric_names = self._normalize_metric_names(metric_names)
        self.metric_params = self._normalize_metric_params(metric_params or {})
        self.fail_fast = fail_fast

    def score(self, ground_truth: object, estimated: object) -> dict[str, float | None]:
        """Return run-record compatible metric values for one run.

        The returned dictionary uses every canonical key in ``METRIC_KEYS``.
        Metrics that were not selected, or metrics that failed with
        ``fail_fast=False``, are represented as ``None``.
        """
        ground_truth_array, estimated_array = prepare_metric_inputs(ground_truth, estimated)
        results: dict[str, float | None] = {}

        for metric_name in METRIC_KEYS:
            if metric_name not in self.metric_names:
                results[metric_name] = None
                continue

            spec = METRIC_SPECS[metric_name]
            params = self.metric_params.get(metric_name, {})

            try:
                metric_result = spec.function.compute(
                    ground_truth_array,
                    estimated_array,
                    **params,
                )
            except Exception:
                if self.fail_fast:
                    raise
                results[metric_name] = None
            else:
                results[metric_name] = float(metric_result.value)

        return results

    @staticmethod
    def _normalize_metric_names(metric_names: list[str] | None) -> tuple[str, ...]:
        """Convert requested names and aliases into canonical registry keys."""
        if metric_names is None:
            return METRIC_KEYS

        normalized_names = []
        for metric_name in metric_names:
            normalized_name = METRIC_ALIASES.get(metric_name, metric_name)
            if normalized_name not in METRIC_SPECS:
                msg = f"Unknown metric: {metric_name}"
                raise KeyError(msg)
            normalized_names.append(normalized_name)

        return tuple(dict.fromkeys(normalized_names))

    @staticmethod
    def _normalize_metric_params(
        metric_params: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        """Normalize metric parameter keys so aliases share one config path."""
        normalized_params = {}
        for metric_name, params in metric_params.items():
            normalized_name = METRIC_ALIASES.get(metric_name, metric_name)
            if normalized_name not in METRIC_SPECS:
                msg = f"Unknown metric: {metric_name}"
                raise KeyError(msg)
            normalized_params[normalized_name] = params

        return normalized_params
