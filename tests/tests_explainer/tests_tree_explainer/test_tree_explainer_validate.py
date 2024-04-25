"""This test module contains all tests for the validation functions of the tree explainer
implementation."""

import copy

import numpy as np
import pytest

from shapiq import safe_isinstance
from shapiq.explainer.tree.validation import validate_tree_model


def test_validate_model(dt_clf_model, dt_reg_model, rf_reg_model, rf_clf_model):
    """Test the validation of the model."""
    class_path_str = ["shapiq.explainer.tree.base.TreeModel"]
    # sklearn dt models are supported
    tree_model = validate_tree_model(dt_clf_model)
    assert safe_isinstance(tree_model, class_path_str)
    tree_model = validate_tree_model(dt_reg_model)
    assert safe_isinstance(tree_model, class_path_str)
    # sklearn rf models are supported
    tree_model = validate_tree_model(rf_clf_model)
    for tree in tree_model:
        assert safe_isinstance(tree, class_path_str)
    tree_model = validate_tree_model(rf_reg_model)
    for tree in tree_model:
        assert safe_isinstance(tree, class_path_str)

    # test the unsupported model
    with pytest.raises(TypeError):
        validate_tree_model("unsupported_model")
