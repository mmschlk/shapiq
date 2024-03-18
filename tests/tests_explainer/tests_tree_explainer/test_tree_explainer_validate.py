"""This test module contains all tests for the validation functions of the tree explainer
implementation."""
import pytest

from shapiq import safe_isinstance
from shapiq.explainer.tree.validation import _validate_model


def test_validate_model(dt_clf_model, dt_reg_model):
    """Test the validation of the model."""
    class_path_str = ["explainer.tree.base.TreeModel"]
    # sklearn dt models are supported
    tree_model = _validate_model(dt_clf_model)
    assert safe_isinstance(tree_model, class_path_str)
    tree_model = _validate_model(dt_reg_model)
    assert safe_isinstance(tree_model, class_path_str)

    # finally, test the unsupported model
    with pytest.raises(TypeError):
        _validate_model("unsupported_model")
