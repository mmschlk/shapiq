"""This test module contains all tests for the validation functions of the tree explainer
implementation."""

import pytest

from shapiq import safe_isinstance
from shapiq.explainer.tree.validation import SUPPORTED_MODELS, validate_tree_model
from tests.conftest import TREE_MODEL_FIXTURES


def test_validate_model(dt_clf_model, dt_reg_model, rf_reg_model, rf_clf_model, if_clf_model):
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
    # sklearn isolation forest is supported
    tree_model = validate_tree_model(if_clf_model)
    for tree in tree_model:
        assert safe_isinstance(tree, class_path_str)

    # test the unsupported model
    with pytest.raises(TypeError):
        validate_tree_model("unsupported_model")


@pytest.mark.external_libraries
@pytest.mark.parametrize("model_fixture, model_class", TREE_MODEL_FIXTURES)
def test_validate_model_fixtures(model_fixture, model_class, background_reg_data, request):
    if model_class not in SUPPORTED_MODELS:
        return
    else:
        model = request.getfixturevalue(model_fixture)
        class_path_str = ["shapiq.explainer.tree.base.TreeModel"]
        tree_model = validate_tree_model(model)
        if type(tree_model) is not list:
            tree_model = [tree_model]
        for tree in tree_model:
            assert safe_isinstance(tree, class_path_str)
