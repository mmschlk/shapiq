"""Protocol tests for all imputers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from shapiq.imputer import (
    BaselineImputer,
    GaussianCopulaImputer,
    GaussianImputer,
    GenerativeConditionalImputer,
    MarginalImputer,
    TabPFNImputer,
)

# ---------------------------------------------------------------------------
# Skip markers (local copies to keep this file self-contained for now)
# ---------------------------------------------------------------------------

skip_if_no_xgboost = pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None, reason="xgboost not installed"
)
skip_if_no_tabpfn = pytest.mark.skipif(
    importlib.util.find_spec("tabpfn") is None, reason="tabpfn not installed"
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_DATA = _RNG.normal(size=(30, 5))
_Y = _DATA[:, 0] + 0.5 * _DATA[:, 1]


def _simple_model(X: np.ndarray) -> np.ndarray:
    return X.sum(axis=1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IMPUTER_CONFIGS = [
    {"cls": MarginalImputer, "kwargs": {"sample_size": 10}},
    {"cls": BaselineImputer, "kwargs": {}},
    {"cls": GaussianImputer, "kwargs": {"sample_size": 10}},
    {"cls": GaussianCopulaImputer, "kwargs": {"sample_size": 10}},
    pytest.param(
        {
            "cls": GenerativeConditionalImputer,
            "kwargs": {"sample_size": 5, "conditional_budget": 16},
        },
        marks=[skip_if_no_xgboost, pytest.mark.slow],
    ),
]


def _imputer_id(config: dict) -> str:
    return config["cls"].__name__


# ===================================================================
# Protocol
# ===================================================================


@pytest.mark.parametrize("config", IMPUTER_CONFIGS, ids=_imputer_id)
class TestImputerProtocol:
    """Contract checks for all imputers."""

    def test_fit_and_call(self, config):
        """fit() + __call__() produces array of correct length."""
        imputer = config["cls"](
            model=_simple_model, data=_DATA.copy(), random_state=42, **config["kwargs"]
        )
        imputer.fit(_DATA[0])

        # All features present
        coalition_all = np.ones((1, 5), dtype=bool)
        result = imputer(coalition_all)
        assert result.shape == (1,)

        # No features present
        coalition_none = np.zeros((1, 5), dtype=bool)
        result = imputer(coalition_none)
        assert result.shape == (1,)

    def test_multiple_coalitions(self, config):
        """Handles batch of coalitions."""
        imputer = config["cls"](
            model=_simple_model, data=_DATA.copy(), random_state=42, **config["kwargs"]
        )
        imputer.fit(_DATA[0])

        coalitions = np.eye(5, dtype=bool)  # one feature at a time
        result = imputer(coalitions)
        assert result.shape == (5,)

    def test_full_coalition_approximates_prediction(self, config):
        """With all features present, result should be close to model prediction."""
        imputer = config["cls"](
            model=_simple_model, data=_DATA.copy(), random_state=42, **config["kwargs"]
        )
        x = _DATA[0]
        imputer.fit(x)

        coalition_all = np.ones((1, 5), dtype=bool)
        result = float(imputer(coalition_all)[0])
        expected = float(_simple_model(x.reshape(1, -1))[0])

        assert result == pytest.approx(expected, abs=0.5)


# ===================================================================
# TabPFNImputer — different constructor signature (x_train, y_train)
# ===================================================================


@skip_if_no_tabpfn
@pytest.mark.slow
class TestTabPFNImputer:
    """TabPFNImputer uses the Remove-and-Contextualize paradigm.

    Unlike other imputers it takes (x_train, y_train) and a predict_function, so it
    cannot participate in ``IMPUTER_CONFIGS`` but the same contract checks apply.
    """

    def _make_imputer(self):
        from tabpfn import TabPFNRegressor

        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(30, 5))
        y_train = x_train[:, 0] + 0.5 * x_train[:, 1]
        model = TabPFNRegressor(n_estimators=1)
        # TabPFN downloads weights from HuggingFace on first fit — skip if offline.
        try:
            model.fit(x_train, y_train)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"TabPFN model unavailable: {exc}")

        def predict_fn(mdl, X):
            return mdl.predict(X)

        return TabPFNImputer(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_train,
            predict_function=predict_fn,
        ), x_train

    def test_fit_and_call(self):
        imputer, x_train = self._make_imputer()
        imputer.fit(x_train[0])

        coalition_all = np.ones((1, 5), dtype=bool)
        assert imputer(coalition_all).shape == (1,)

        coalition_none = np.zeros((1, 5), dtype=bool)
        assert imputer(coalition_none).shape == (1,)

    def test_multiple_coalitions(self):
        imputer, x_train = self._make_imputer()
        imputer.fit(x_train[0])
        coalitions = np.eye(5, dtype=bool)
        assert imputer(coalitions).shape == (5,)
