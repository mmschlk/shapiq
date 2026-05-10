import unittest
import numpy as np

from metrics import METRICS


class MetricsTestCase(unittest.TestCase):

    def test_mse_is_zero_for_equal_values(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([1.0, 2.0, 3.0])

        metric = METRICS["mse"]
        score = metric.compute(ground_truth, estimated)

        result = metric.compute(ground_truth, estimated)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.metric_name, "mse")
        self.assertFalse(result.higher_is_better)

    def test_normalized_mse_is_zero_for_equal_values(self):
        ground_truth = np.array([1.0, 2.0, 3.0])
        estimated = np.array([1.0, 2.0, 3.0])

        metric = METRICS["normalized_mse"]

        score = metric.compute(self,ground_truth, estimated)

        result = metric.compute(ground_truth, estimated)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.metric_name, "normalized_mse")
        self.assertFalse(result.higher_is_better)

if __name__ == "__main__":
    unittest.main()