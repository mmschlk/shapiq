"""All tests that check if the prediction functions are extracted and selected correctly."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from shapiq.explainer.utils import get_predict_function_and_model_type


class TestSklearnAPI:
    """All tests that check if prediction functions are extracted correctly for sklearn-like models."""

    _dummy_input = np.zeros((1, 3))
    """A dummy input for testing purposes to be used in the tests."""

    def _dummy_predict(self, x: np.ndarray) -> np.ndarray:
        """A dummy predict function that returns zeros."""
        return np.array([0.0] * x.shape[0])

    def _assert_uses(self, model, function_name: str, monkeypatch) -> None:
        """Assert that the model's prediction function is the shapiq function of that name."""
        flag_is_used = False

        def mock_predict(model, x):
            nonlocal flag_is_used
            flag_is_used = True
            return self._dummy_predict(x)

        monkeypatch.setattr(f"shapiq.explainer.utils.{function_name}", mock_predict)
        predict_function, _ = get_predict_function_and_model_type(model)
        if not isinstance(predict_function, RuntimeError):
            _ = predict_function(model, self._dummy_input)
        assert flag_is_used, f"The {function_name} prediction function was not used."

    @pytest.mark.parametrize("model_fixture", ["tabpfn_clf_model"])
    def test_uses_predict_logit(self, model_fixture, request, monkeypatch) -> None:
        """Models that should use the logits as the prediction function."""
        model = request.getfixturevalue(model_fixture)
        self._assert_uses(model, "predict_logits", monkeypatch)

    @pytest.mark.parametrize(
        "model_fixture", ["lr_clf_model", "gb_clf_model", "hist_gb_clf_model", "bin_svc_model"]
    )
    def test_uses_decision_function(self, model_fixture, request, monkeypatch) -> None:
        """Classifiers with a decision function are explained in margin (log-odds) space."""
        model = request.getfixturevalue(model_fixture)
        assert hasattr(model, "decision_function")
        self._assert_uses(model, "predict_decision_function", monkeypatch)

    def test_uses_decision_function_isolation_forest(self, if_clf_model, monkeypatch) -> None:
        """An isolation forest is explained by its anomaly score, not by its -1/1 labels."""
        self._assert_uses(if_clf_model, "predict_decision_function", monkeypatch)

    @pytest.mark.parametrize("model_fixture", ["rf_clf_model", "dt_clf_model"])
    def test_uses_predict_proba(self, model_fixture, request, monkeypatch) -> None:
        """Classifiers without a decision function are explained by their probabilities."""
        model = request.getfixturevalue(model_fixture)
        assert not hasattr(model, "decision_function")
        self._assert_uses(model, "predict_proba", monkeypatch)

    @pytest.mark.parametrize("model_fixture", ["lr_reg_model", "rf_reg_model"])
    def test_uses_predict(self, model_fixture, request, monkeypatch) -> None:
        """Regressors are explained by their predictions."""
        model = request.getfixturevalue(model_fixture)
        self._assert_uses(model, "predict", monkeypatch)

    def test_extraction_order(self, monkeypatch) -> None:
        """Logits beat the decision function, which beats probabilities, which beat predictions."""

        class Model:
            def predict(self, x):
                return np.full(x.shape[0], 1.0)

            def predict_proba(self, x):
                return np.full((x.shape[0], 2), 2.0)

            def decision_function(self, x):
                return np.full(x.shape[0], 3.0)

            def predict_logits(self, x):
                return np.full((x.shape[0], 2), 4.0)

        def _value(model) -> float:
            predict_function, _ = get_predict_function_and_model_type(model)
            return float(predict_function(model, self._dummy_input)[0])

        model = Model()
        assert _value(model) == 4.0
        monkeypatch.delattr(Model, "predict_logits")
        assert _value(model) == 3.0
        monkeypatch.delattr(Model, "decision_function")
        assert _value(model) == 2.0
        monkeypatch.delattr(Model, "predict_proba")
        assert _value(model) == 1.0


class TestDecisionFunctionClassIndex:
    """Tests of the class index with the decision function of binary and multiclass classifiers."""

    def test_binary_margin(self, background_clf_dataset_binary) -> None:
        """The binary decision function is the margin of class 1 and its negative for class 0."""
        X, y = background_clf_dataset_binary
        model = LogisticRegression(random_state=42, max_iter=200).fit(X, y)
        margins = model.decision_function(X)
        assert margins.ndim == 1

        predict_function, _ = get_predict_function_and_model_type(model)
        np.testing.assert_allclose(predict_function(model, X), margins)
        predict_function, _ = get_predict_function_and_model_type(model, class_index=1)
        np.testing.assert_allclose(predict_function(model, X), margins)
        predict_function, _ = get_predict_function_and_model_type(model, class_index=0)
        np.testing.assert_allclose(predict_function(model, X), -margins)

        # the margins are the log-odds of the classes
        proba = model.predict_proba(X)
        np.testing.assert_allclose(margins, np.log(proba[:, 1] / proba[:, 0]))

    def test_binary_margin_invalid_class_index(self, background_clf_dataset_binary) -> None:
        """A class index other than 0 or 1 cannot be selected from a single margin."""
        X, y = background_clf_dataset_binary
        model = LogisticRegression(random_state=42, max_iter=200).fit(X, y)
        predict_function, _ = get_predict_function_and_model_type(model, class_index=2)
        with pytest.raises(ValueError, match="class_index=2"):
            predict_function(model, X)

    def test_multiclass_margin(self, lr_clf_model, background_clf_data) -> None:
        """The class index selects the column of a multiclass decision function."""
        margins = lr_clf_model.decision_function(background_clf_data)
        assert margins.ndim == 2
        for class_index in range(margins.shape[1]):
            predict_function, _ = get_predict_function_and_model_type(
                lr_clf_model, class_index=class_index
            )
            np.testing.assert_allclose(
                predict_function(lr_clf_model, background_clf_data), margins[:, class_index]
            )

    def test_score_without_classes_ignores_class_index(self, if_clf_model, if_clf_dataset) -> None:
        """A single score of a model without classes (an anomaly score) is returned as is."""
        X, _ = if_clf_dataset
        scores = if_clf_model.decision_function(X)
        assert not hasattr(if_clf_model, "classes_")
        for class_index in (None, 0, 1, 2):
            predict_function, _ = get_predict_function_and_model_type(
                if_clf_model, class_index=class_index
            )
            np.testing.assert_allclose(predict_function(if_clf_model, X), scores)
