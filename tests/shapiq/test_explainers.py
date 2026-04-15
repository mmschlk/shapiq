"""Protocol and special-case tests for explainers."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.explainer.agnostic import AgnosticExplainer
from shapiq.explainer.tabular import TabularExplainer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame

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
