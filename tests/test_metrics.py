from __future__ import annotations

import math
import unittest

import numpy as np

from metrics import METRIC_KEYS, METRICS, Scorer
from runner.aggregator import aggregate_metric_values, aggregate_run_records


class MetricsTestCase(unittest.TestCase):
    def test_mse_is_zero_for_equal_values(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([1.0, 2.0, 3.0])

        result = METRICS["mse"].compute(ground_truth, estimated)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.metric_name, "mse")
        self.assertFalse(result.higher_is_better)

    def test_normalized_mse_uses_run_record_name(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([1.0, 2.0, 3.0])

        result = METRICS["mse_normalized"].compute(ground_truth, estimated)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.metric_name, "mse_normalized")
        self.assertFalse(result.higher_is_better)

    def test_normalized_mse_alias_remains_available(self):
        self.assertIs(METRICS["normalized_mse"], METRICS["mse_normalized"])

    def test_r2_is_one_for_perfect_prediction(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([1.0, 2.0, 3.0])

        result = METRICS["r2"].compute(ground_truth, estimated)

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.metric_name, "r2")
        self.assertTrue(result.higher_is_better)

    def test_r2_can_be_negative_for_poor_prediction(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([3.0, 2.0, 1.0])

        result = METRICS["r2"].compute(ground_truth, estimated)

        self.assertLess(result.value, 0.0)

    def test_r2_returns_nan_for_constant_ground_truth(self):
        ground_truth = np.array([2.0, 2.0, 2.0])
        estimated = np.array([2.0, 2.0, 2.0])

        result = METRICS["r2"].compute(ground_truth, estimated)

        self.assertTrue(math.isnan(result.value))


class ScorerTestCase(unittest.TestCase):
    def test_computes_selected_metrics_and_preserves_keys(self):
        scorer = Scorer(metric_names=["mse", "normalized_mse", "r2"])

        scores = scorer.score(
            ground_truth=np.array([1.0, 2.0, 3.0]),
            estimated=np.array([1.0, 2.0, 3.0]),
        )

        self.assertEqual(tuple(scores.keys()), METRIC_KEYS)
        self.assertEqual(scores["mse"], 0.0)
        self.assertEqual(scores["mse_normalized"], 0.0)
        self.assertEqual(scores["r2"], 1.0)
        self.assertIsNone(scores["spearman"])

    def test_supports_metric_params_for_precision_at_k(self):
        scorer = Scorer(
            metric_names=["precision_at_k"],
            metric_params={"precision_at_k": {"k": 2}},
        )

        scores = scorer.score(
            ground_truth=np.array([10.0, 9.0, 1.0]),
            estimated=np.array([1.0, 9.0, 10.0]),
        )

        self.assertEqual(scores["precision_at_k"], 0.5)

    def test_handles_metric_failure_without_crashing_by_default(self):
        scorer = Scorer(
            metric_names=["precision_at_k"],
            metric_params={"precision_at_k": {"k": 0}},
        )

        scores = scorer.score(
            ground_truth=np.array([1.0, 2.0, 3.0]),
            estimated=np.array([1.0, 2.0, 3.0]),
        )

        self.assertIsNone(scores["precision_at_k"])

    def test_raises_metric_failure_when_fail_fast_is_true(self):
        scorer = Scorer(
            metric_names=["precision_at_k"],
            metric_params={"precision_at_k": {"k": 0}},
            fail_fast=True,
        )

        with self.assertRaises(ValueError):
            scorer.score(
                ground_truth=np.array([1.0, 2.0, 3.0]),
                estimated=np.array([1.0, 2.0, 3.0]),
            )

    def test_shape_mismatch_raises_clear_error(self):
        scorer = Scorer(metric_names=["mse"])

        with self.assertRaisesRegex(ValueError, "same shape"):
            scorer.score(
                ground_truth=np.array([1.0, 2.0, 3.0]),
                estimated=np.array([1.0, 2.0]),
            )


class ScorerAggregatorIntegrationTestCase(unittest.TestCase):
    def test_scored_run_records_can_be_aggregated(self):
        scorer = Scorer(metric_names=["mse", "r2"])
        first_metrics = scorer.score(
            ground_truth=np.array([1.0, 2.0, 3.0]),
            estimated=np.array([1.0, 2.0, 3.0]),
        )
        second_metrics = scorer.score(
            ground_truth=np.array([1.0, 2.0, 3.0]),
            estimated=np.array([1.0, 2.0, 4.0]),
        )

        records = [
            self._run_record(first_metrics, runtime_seconds=1.0),
            self._run_record(second_metrics, runtime_seconds=3.0),
        ]

        aggregated_metrics = aggregate_metric_values(records)
        aggregated_record = aggregate_run_records(records)

        self.assertIn("mse", aggregated_metrics)
        self.assertIn("r2", aggregated_metrics)
        self.assertAlmostEqual(aggregated_metrics["mse"], 1 / 6)
        self.assertAlmostEqual(aggregated_record["metrics"]["r2"], 0.75)
        self.assertEqual(aggregated_record["runtime_seconds"], 2.0)

    @staticmethod
    def _run_record(metrics, runtime_seconds):
        return {
            "run_id": "run-id",
            "game_name": "DummyGame",
            "game_id": "dummy-game-id",
            "game_params": {"n_players": 3},
            "n_players": 3,
            "approximator_name": "DummyApproximator",
            "approximator_params": {},
            "shapiq_version": "0.0.0",
            "index": "SV",
            "max_order": 1,
            "budget": 8,
            "approx_seed": 1,
            "ground_truth_method": "ExactComputer",
            "run_failed": False,
            "error_message": None,
            "metrics": metrics,
            "runtime_seconds": runtime_seconds,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "hardware": {},
            "notes": "",
        }


if __name__ == "__main__":
    unittest.main()
