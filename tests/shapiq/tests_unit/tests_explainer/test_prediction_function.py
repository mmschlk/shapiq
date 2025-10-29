"""All tests that check if the prediction functions are extracted and selected correctly."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.explainer.utils import get_predict_function_and_model_type


class TestSklearnAPI:
    """All tests that check if prediction functions are extracted correctly for sklearn-like models."""

    _dummy_input = np.zeros((1, 3))
    """A dummy input for testing purposes to be used in the tests."""

    def _dummy_predict(self, x: np.ndarray) -> np.ndarray:
        """A dummy predict function that returns zeros."""
        return np.array([0.0] * x.shape[0])

    @pytest.mark.parametrize("model_fixture", ["tabpfn_clf_model"])
    def test_uses_predict_logit(self, model_fixture, request, monkeypatch) -> None:
        """Models that should use the logits as the prediction function."""
        flag_is_used = False

        def mock_predict_logits(model, x):
            nonlocal flag_is_used
            flag_is_used = True
            return self._dummy_predict(x)

        monkeypatch.setattr("shapiq.explainer.utils.predict_logits", mock_predict_logits)
        model = request.getfixturevalue(model_fixture)
        predict_function, _ = get_predict_function_and_model_type(model)
        if not isinstance(predict_function, RuntimeError):
            _ = predict_function(model, self._dummy_input)
        assert flag_is_used, "The logits prediction function was not used."
