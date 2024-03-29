"""This test module contains all tests for the validation functions of the tree explainer
implementation."""
import copy

import pytest
import numpy as np

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


def test_validate_output_types_parameters(dt_clf_model, dt_clf_model_tree_model):
    """This test checks weather the correct output types are validated.

    This test does not check if the conversion of the output types is semantically correct. This is
    tested in the next test.

    """
    class_path_str = ["shapiq.explainer.tree.base.TreeModel"]

    # test with invalid output type
    with pytest.raises(ValueError):
        validate_tree_model(dt_clf_model, output_type="invalid_output_type")

    # test with 'raw' output type
    tree_model = validate_tree_model(dt_clf_model, output_type="raw")
    assert safe_isinstance(tree_model, class_path_str)

    # test with 'probability' output type (probability from probability)
    tree_model = validate_tree_model(dt_clf_model, output_type="probability")
    assert safe_isinstance(tree_model, class_path_str)

    # test with 'logit' output type (logit from probability)
    tree_model = validate_tree_model(dt_clf_model, output_type="logit")
    assert safe_isinstance(tree_model, class_path_str)

    from shapiq.explainer.tree.base import convert_tree_output_type

    # test with 'probability from 'logit' output type
    tree_model_logit = copy.deepcopy(dt_clf_model_tree_model)
    tree_model_logit.original_output_type = "logit"
    # manually change the values to logit from probabilities
    tree_model_logit.values = np.log(tree_model_logit.values / (1 - tree_model_logit.values))
    tree_model_logit, _ = convert_tree_output_type(tree_model_logit, output_type="probability")
    assert safe_isinstance(tree_model_logit, class_path_str)

    # test edge cases
    tree_model, _ = convert_tree_output_type(dt_clf_model_tree_model, output_type="raw")
    assert safe_isinstance(tree_model, class_path_str)
