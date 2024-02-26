"""This test module collects all tests for the utility functions of the tree explainer."""
import numpy as np

from shapiq.explainer.tree.utils import (
    compute_empty_prediction,
    get_conditional_sample_weights,
)


def test_compute_empty_prediction():
    """Test the compute_empty_prediction function."""
    values = np.asarray([100, 200, 300, 400, 500])
    sample_weights = np.asarray([1.0, 0.5, 0.25, 0.25, 0.5])
    is_leaf = np.asarray([False, False, True, True, True])
    leaf_values = values[is_leaf]
    leaf_sample_weights = sample_weights[is_leaf]
    empty_prediction = compute_empty_prediction(leaf_values, leaf_sample_weights)
    assert empty_prediction == 425.0


def test_conditional_sample_weights():
    """Test the conditional sample utils function."""
    par_arr = [-1, 0, 1, 1, 0, 4, 4]
    par_arr = np.asarray(par_arr)
    count_arr = [100, 70, 50, 20, 30, 15, 15]
    count_arr = np.asarray(count_arr)

    weights = get_conditional_sample_weights(parent_array=par_arr, sample_count=count_arr)
    assert weights[0] == 1.0
    assert weights[2] == 50 / 70
    assert weights[5] == 0.5
