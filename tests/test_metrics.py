from __future__ import annotations

import unittest

import numpy as np
from runner.aggregator import aggregate_metric_values, aggregate_run_records

from leaderboard.metrics import METRIC_KEYS, METRICS, Scorer
from leaderboard.metrics.evaluator import compute_all_metrics
from leaderboard.metrics.utils import prepare_metric_inputs
from leaderboard.runner.aggregator import aggregate_metric_values, aggregate_run_records
from shapiq import InteractionValues

EXPECTED_METRIC_KEYS = (
    "mse",
    "mae",
    "mse_normalized",
    "spearman",
    "kendall_tau",
    "precision_at_k",
)


def interaction_values(
    values_by_interaction: dict[tuple[int, ...], float],
    n_players: int = 20,
) -> InteractionValues:
    interactions = list(values_by_interaction)
    return InteractionValues(
        values=np.array(
            [values_by_interaction[interaction] for interaction in interactions], dtype=float
        ),
        index="SV",
        max_order=max((len(interaction) for interaction in interactions), default=0),
        n_players=n_players,
        min_order=0,
        interaction_lookup={interaction: index for index, interaction in enumerate(interactions)},
    )


from metrics import METRIC_KEYS, METRICS, Scorer


class MetricsTestCase(unittest.TestCase):
    def test_public_api_exports_required_metrics(self):
        self.assertEqual(METRIC_KEYS, EXPECTED_METRIC_KEYS)
        self.assertEqual(
            tuple(compute_all_metrics(np.array([1.0]), np.array([1.0])).keys()), METRIC_KEYS
        )
        self.assertIs(METRICS["normalized_mse"], METRICS["mse_normalized"])
        self.assertNotIn("r2", METRICS)

    def test_mse_is_zero_for_equal_values(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([1.0, 2.0, 3.0])

        result = METRICS["mse"].compute(ground_truth, estimated)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.metric_name, "mse")
        self.assertFalse(result.higher_is_better)

    def test_mae_uses_absolute_error(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([2.0, 0.0, 6.0])

        result = METRICS["mae"].compute(ground_truth, estimated)

        self.assertEqual(result.value, 2.0)
        self.assertEqual(result.metric_name, "mae")
        self.assertFalse(result.higher_is_better)

    def test_normalized_mse_uses_run_record_name(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([1.0, 2.0, 3.0])

        result = METRICS["mse_normalized"].compute(ground_truth, estimated)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.metric_name, "mse_normalized")
        self.assertFalse(result.higher_is_better)

    def test_kendall_tau_handles_nan_as_zero(self):
        ground_truth = np.array([1.0, 1.0, 1.0])
        estimated = np.array([2.0, 2.0, 2.0])

        result = METRICS["kendall_tau"].compute(ground_truth, estimated)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.metric_name, "kendall_tau")
        self.assertTrue(result.higher_is_better)

    def test_precision_at_k_uses_absolute_top_k_overlap(self):
        ground_truth = np.array([10.0, -9.0, 1.0])
        estimated = np.array([1.0, -9.0, 10.0])

        result = METRICS["precision_at_k"].compute(ground_truth, estimated, k=2)

        self.assertEqual(result.value, 0.5)
        self.assertEqual(result.metric_name, "precision_at_k")
        self.assertTrue(result.higher_is_better)


class PrepareMetricInputsTestCase(unittest.TestCase):
    def test_numpy_arrays_keep_existing_shape_behavior(self):
        ground_truth = np.array([[1.0, 2.0]])
        estimated = np.array([[1.5, 2.5]])

        prepared_ground_truth, prepared_estimated = prepare_metric_inputs(ground_truth, estimated)

        np.testing.assert_array_equal(prepared_ground_truth, ground_truth)
        np.testing.assert_array_equal(prepared_estimated, estimated)

    def test_interaction_values_ignore_empty_key_and_align_union(self):
        ground_truth = interaction_values({(): 100.0, (0,): 1.0, (1,): 2.0})
        estimated = interaction_values({(): -100.0, (1,): 3.0, (2,): 4.0})

        prepared_ground_truth, prepared_estimated = prepare_metric_inputs(ground_truth, estimated)

        np.testing.assert_array_equal(prepared_ground_truth, np.array([1.0, 2.0, 0.0]))
        np.testing.assert_array_equal(prepared_estimated, np.array([0.0, 3.0, 4.0]))

    def test_spearman_uses_existing_interaction_keys_without_powerset(self):
        ground_truth = interaction_values({(): 999.0, (0,): 1.0, (19,): 2.0}, n_players=20)
        estimated = interaction_values({(): -999.0, (0,): 1.0, (19,): 3.0}, n_players=20)

        scores = Scorer(metric_names=["spearman"]).score(ground_truth, estimated)

        self.assertEqual(scores["spearman"], 0.9999999999999999)

    def test_precision_at_k_ignores_empty_interaction_values_key(self):
        ground_truth = interaction_values({(): 1000.0, (0,): 5.0, (1,): 4.0})
        estimated = interaction_values({(): 1000.0, (0,): 5.0, (2,): 4.0})

        scores = Scorer(
            metric_names=["precision_at_k"], metric_params={"precision_at_k": {"k": 2}}
        ).score(
            ground_truth,
            estimated,
        )

        self.assertEqual(scores["precision_at_k"], 0.5)

    def test_precision_at_k_compares_interaction_keys_not_values(self):
        ground_truth = interaction_values({(0,): 5.0, (1,): 4.0})
        estimated = interaction_values({(0,): 50.0, (2,): 4.0})

        scores = Scorer(
            metric_names=["precision_at_k"], metric_params={"precision_at_k": {"k": 2}}
        ).score(
            ground_truth,
            estimated,
        )

        self.assertEqual(scores["precision_at_k"], 0.5)


class ScorerTestCase(unittest.TestCase):
    def test_computes_selected_metrics_and_preserves_keys(self):
        scorer = Scorer(metric_names=["mse", "normalized_mse", "mae"])

        scores = scorer.score(
            ground_truth=np.array([1.0, 2.0, 3.0]),
            estimated=np.array([1.0, 2.0, 3.0]),
        )

        self.assertEqual(tuple(scores.keys()), METRIC_KEYS)
        self.assertEqual(scores["mse"], 0.0)
        self.assertEqual(scores["mse_normalized"], 0.0)
        self.assertEqual(scores["mae"], 0.0)
        self.assertNotIn("normalized_mse", scores)
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

    def test_unknown_r2_metric_raises_key_error(self):
        with self.assertRaises(KeyError):
            Scorer(metric_names=["r2"])


class ScorerAggregatorIntegrationTestCase(unittest.TestCase):
    def test_scored_run_records_can_be_aggregated(self):
        scorer = Scorer(metric_names=["mse", "mae"])
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
        self.assertIn("mae", aggregated_metrics)
        self.assertAlmostEqual(aggregated_metrics["mse"], 1 / 6)
        self.assertAlmostEqual(aggregated_record["metrics"]["mae"], 1 / 6)
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
