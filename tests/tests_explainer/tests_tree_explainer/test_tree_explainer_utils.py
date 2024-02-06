"""This test module collects all tests for the utility functions of the tree explainer."""
import numpy as np

from shapiq.explainer.tree.utils import compute_empty_prediction


def test_compute_empty_prediction():
    """Test the compute_empty_prediction function."""
    values = np.asarray([100, 200, 300, 400, 500])
    sample_weights = np.asarray([1.0, 0.5, 0.25, 0.25, 0.5])
    is_leaf = np.asarray([False, False, True, True, True])
    leaf_values = values[is_leaf]
    leaf_sample_weights = sample_weights[is_leaf]
    empty_prediction = compute_empty_prediction(leaf_values, leaf_sample_weights)
    assert empty_prediction == 425.0
