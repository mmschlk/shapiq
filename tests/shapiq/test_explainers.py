"""Protocol and special-case tests for explainers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from shapiq.explainer.agnostic import AgnosticExplainer
from shapiq.explainer.product_kernel import ProductKernelExplainer
from shapiq.explainer.tabular import TabularExplainer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame

skip_if_no_tabpfn = pytest.mark.skipif(
    importlib.util.find_spec("tabpfn") is None, reason="tabpfn not installed"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_model(X: np.ndarray) -> np.ndarray:
    """Simple linear model: sum of features."""
    return X.sum(axis=1)


def _make_background_data(n_samples: int = 30, n_features: int = 5) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_samples, n_features))


# ===================================================================
# TabularExplainer protocol
# ===================================================================


TABULAR_CONFIGS = [
    {"index": "SV", "max_order": 1, "approximator": "auto"},
    {"index": "k-SII", "max_order": 2, "approximator": "auto"},
    {"index": "FSII", "max_order": 2, "approximator": "regression"},
    {"index": "SV", "max_order": 1, "approximator": "permutation"},
    {"index": "k-SII", "max_order": 2, "approximator": "montecarlo"},
]


def _tabular_id(config: dict) -> str:
    return f"{config['index']}-order{config['max_order']}-{config['approximator']}"


@pytest.mark.parametrize("config", TABULAR_CONFIGS, ids=_tabular_id)
class TestTabularExplainerProtocol:
    """Contract checks for TabularExplainer."""

    def test_explain_returns_interaction_values(self, config):
        """explain() returns InteractionValues with correct metadata."""
        data = _make_background_data()
        explainer = TabularExplainer(
            model=_linear_model,
            data=data,
            index=config["index"],
            max_order=config["max_order"],
            approximator=config["approximator"],
            random_state=42,
        )
        x = data[0].reshape(1, -1)
        result = explainer.explain(x, budget=2**5)

        assert isinstance(result, InteractionValues)
        assert result.index == config["index"]
        assert result.max_order == config["max_order"]

    def test_efficiency(self, config):
        """sum(values) approximates the prediction for efficient indices."""
        if config["index"] not in {"SV", "k-SII", "FSII"}:
            pytest.skip("Efficiency not guaranteed for this index")
        data = _make_background_data()
        explainer = TabularExplainer(
            model=_linear_model,
            data=data,
            index=config["index"],
            max_order=config["max_order"],
            approximator=config["approximator"],
            random_state=42,
        )
        x = data[0].reshape(1, -1)
        result = explainer.explain(x, budget=2**5)
        prediction = float(_linear_model(x)[0])
        assert float(np.sum(result.values)) == pytest.approx(prediction, abs=0.5)

    def test_reproducible(self, config):
        """Same random_state produces identical explanations."""
        data = _make_background_data()
        kwargs = {
            "model": _linear_model,
            "data": data,
            "index": config["index"],
            "max_order": config["max_order"],
            "approximator": config["approximator"],
        }
        e1 = TabularExplainer(**kwargs, random_state=42)
        e2 = TabularExplainer(**kwargs, random_state=42)
        x = data[0].reshape(1, -1)
        r1 = e1.explain(x, budget=2**5, random_state=42)
        r2 = e2.explain(x, budget=2**5, random_state=42)
        assert np.allclose(r1.values, r2.values)


# ===================================================================
# AgnosticExplainer
# ===================================================================


class TestAgnosticExplainer:
    """AgnosticExplainer wraps a Game or callable directly."""

    def test_explain_with_game(self):
        game = DummyGame(n=5, interaction=(0, 1))
        explainer = AgnosticExplainer(game=game, index="k-SII", max_order=2, random_state=42)
        result = explainer.explain(budget=100)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 5

    def test_explain_with_callable(self):
        def my_game(coalitions):
            return coalitions.sum(axis=1).astype(float)

        explainer = AgnosticExplainer(
            game=my_game, n_players=4, index="SV", max_order=1, random_state=42
        )
        result = explainer.explain(budget=50)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 4


# ===================================================================
# TabularExplainer special cases
# ===================================================================


class TestTabularExplainerValidation:
    """Input validation and edge cases."""

    def test_invalid_index_raises(self):
        data = _make_background_data()
        with pytest.raises(ValueError):
            TabularExplainer(model=_linear_model, data=data, index="INVALID", max_order=2)

    def test_invalid_approximator_raises(self):
        data = _make_background_data()
        with pytest.raises(ValueError):
            TabularExplainer(model=_linear_model, data=data, approximator="not_real")

    def test_sv_with_high_order_warns(self):
        data = _make_background_data()
        with pytest.warns(UserWarning):
            TabularExplainer(model=_linear_model, data=data, index="SV", max_order=2)

    def test_explain_X_batch(self):
        """explain_X returns a list of InteractionValues."""
        data = _make_background_data()
        explainer = TabularExplainer(
            model=_linear_model, data=data, index="SV", max_order=1, random_state=42
        )
        results = explainer.explain_X(data[:3], budget=2**5)
        assert len(results) == 3
        assert all(isinstance(r, InteractionValues) for r in results)


# ===================================================================
# ProductKernelExplainer
# ===================================================================


def _pk_regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, size=30)
    return X, y


def _pk_classification_data() -> tuple[np.ndarray, np.ndarray]:
    X, y_reg = _pk_regression_data()
    y = (y_reg > np.median(y_reg)).astype(int)
    return X, y


def _svr_model():
    from sklearn.svm import SVR

    X, y = _pk_regression_data()
    model = SVR(kernel="rbf")
    model.fit(X, y)
    return model, X


def _svc_model():
    from sklearn.svm import SVC

    X, y = _pk_classification_data()
    model = SVC(kernel="rbf", probability=True)
    model.fit(X, y)
    return model, X


def _gpr_model():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    X, y = _pk_regression_data()
    model = GaussianProcessRegressor(kernel=RBF(), random_state=42)
    model.fit(X, y)
    return model, X


PRODUCT_KERNEL_MODELS = [
    ("svr", _svr_model, "regression"),
    ("svc", _svc_model, "classification"),
    ("gpr", _gpr_model, "regression"),
]


@pytest.mark.parametrize(
    ("model_name", "model_factory", "task"),
    PRODUCT_KERNEL_MODELS,
    ids=[m[0] for m in PRODUCT_KERNEL_MODELS],
)
class TestProductKernelExplainerProtocol:
    """Contract checks for ProductKernelExplainer across supported model types."""

    def test_explain_returns_interaction_values(self, model_name, model_factory, task):
        model, X = model_factory()
        explainer = ProductKernelExplainer(model=model, max_order=1, min_order=0, index="SV")
        result = explainer.explain(X[0])

        assert isinstance(result, InteractionValues)
        assert result.n_players == X.shape[1]
        assert result.max_order == 1
        assert result.estimated is False

    def test_efficiency_for_regression(self, model_name, model_factory, task):
        """For regression SVR/GPR, sum(values) equals the model prediction."""
        if task != "regression":
            pytest.skip("Efficiency check only well-defined for regression models")
        model, X = model_factory()
        explainer = ProductKernelExplainer(model=model, max_order=1, min_order=0, index="SV")
        result = explainer.explain(X[0])
        prediction = float(model.predict(X[0].reshape(1, -1))[0])
        assert float(np.sum(result.values)) == pytest.approx(prediction, abs=1e-4)

    def test_explain_X_batch(self, model_name, model_factory, task):
        """explain_X returns a list of InteractionValues for each input."""
        model, X = model_factory()
        explainer = ProductKernelExplainer(model=model, max_order=1, min_order=0, index="SV")
        results = explainer.explain_X(X[:3])
        assert len(results) == 3
        assert all(isinstance(r, InteractionValues) for r in results)
        assert all(r.n_players == X.shape[1] for r in results)


class TestProductKernelExplainerValidation:
    """Input validation and edge cases for ProductKernelExplainer."""

    def test_max_order_above_one_raises(self):
        from sklearn.svm import SVR

        X, y = _pk_regression_data()
        model = SVR(kernel="rbf").fit(X, y)
        with pytest.raises(ValueError, match="max_order=1"):
            ProductKernelExplainer(model=model, max_order=2)

    def test_unsupported_model_raises(self):
        from sklearn.linear_model import LinearRegression

        X, y = _pk_regression_data()
        model = LinearRegression().fit(X, y)
        with pytest.raises(TypeError, match="Unsupported model type"):
            ProductKernelExplainer(model=model, max_order=1)

    def test_multiclass_svc_raises(self):
        from sklearn.svm import SVC

        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 5))
        y = rng.integers(0, 3, size=60)
        model = SVC(kernel="rbf", probability=True).fit(X, y)
        with pytest.raises(TypeError, match="binary SVM"):
            ProductKernelExplainer(model=model, max_order=1)


# ===================================================================
# TabPFNExplainer (slow — downloads model weights from HuggingFace)
# ===================================================================


@skip_if_no_tabpfn
@pytest.mark.slow
class TestTabPFNExplainer:
    """TabPFNExplainer wraps TabularExplainer for TabPFN in-context learners."""

    def _make_explainer(self):
        from tabpfn import TabPFNRegressor

        from shapiq.explainer.tabpfn import TabPFNExplainer

        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        y = X[:, 0] + 0.5 * X[:, 1]
        model = TabPFNRegressor(n_estimators=1)
        try:
            model.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"TabPFN model unavailable: {exc}")

        explainer = TabPFNExplainer(
            model=model,
            data=X,
            labels=y,
            index="SV",
            max_order=1,
        )
        return explainer, X

    def test_explain_returns_interaction_values(self):
        explainer, X = self._make_explainer()
        result = explainer.explain(X[0].reshape(1, -1), budget=2**5)
        assert isinstance(result, InteractionValues)
        assert result.index == "SV"
        assert result.max_order == 1
        assert result.n_players == X.shape[1]

    def test_explain_X_batch(self):
        explainer, X = self._make_explainer()
        results = explainer.explain_X(X[:2], budget=2**5)
        assert len(results) == 2
        assert all(isinstance(r, InteractionValues) for r in results)
