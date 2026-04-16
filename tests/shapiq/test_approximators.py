"""Protocol and special-case tests for all approximators."""

from __future__ import annotations

import inspect

import pytest

from shapiq.approximator import (
    SHAPIQ,
    SPEX,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    MSRBiased,
    OwenSamplingSV,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    ProxySHAP,
    RegressionFBII,
    RegressionFSII,
    StratifiedSamplingSV,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame

from .conftest import assert_iv_close

# ---------------------------------------------------------------------------
# Indices where sum(values) == game(N) - game(empty) holds
# ---------------------------------------------------------------------------
EFFICIENT_INDICES = {"SV", "k-SII", "FSII", "STII", "kADD-SHAP"}

# ---------------------------------------------------------------------------
# Registry: one entry per (approximator, index) combo worth testing.
# Each entry must work with a 7-player DummyGame(interaction=(1,2)).
# ---------------------------------------------------------------------------
ALL_APPROXIMATORS = [
    # --- Permutation ---
    {"cls": PermutationSamplingSV, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": PermutationSamplingSII, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": PermutationSamplingSTII, "n": 7, "max_order": 2, "index": "STII", "budget": 100},
    # --- Marginals ---
    {"cls": OwenSamplingSV, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": StratifiedSamplingSV, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    # --- Monte Carlo ---
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "SII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "FSII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "FBII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "STII", "budget": 100},
    {"cls": UnbiasedKernelSHAP, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": SVARM, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": SVARMIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": SVARMIQ, "n": 7, "max_order": 2, "index": "SII", "budget": 100},
    # --- Regression ---
    {"cls": KernelSHAP, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": KernelSHAPIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": InconsistentKernelSHAPIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": kADDSHAP, "n": 7, "max_order": 2, "index": "kADD-SHAP", "budget": 100},
    {"cls": RegressionFSII, "n": 7, "max_order": 2, "index": "FSII", "budget": 100},
    {"cls": RegressionFBII, "n": 7, "max_order": 2, "index": "FBII", "budget": 100},
    # --- Sparse ---
    {"cls": SPEX, "n": 7, "max_order": 2, "index": "FSII", "budget": 300},
    # --- Proxy ---
    {"cls": MSRBiased, "n": 7, "max_order": 2, "index": "SII", "budget": 100},
    {"cls": MSRBiased, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
]


def _approx_id(config: dict) -> str:
    return f"{config['cls'].__name__}-{config['index']}"


def _signature_params(cls: type) -> tuple[set[str], bool]:
    """Return (named params, accepts_var_kwargs) for cls.__init__."""
    params = inspect.signature(cls.__init__).parameters
    named = {name for name in params if name != "self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return named, has_var_kw


def _make_approximator(config: dict, **overrides):
    """Instantiate the approximator, passing only kwargs its signature accepts."""
    cls = config["cls"]
    named, has_var_kw = _signature_params(cls)

    candidate: dict = {"n": config["n"]}
    if "max_order" in named:
        candidate["max_order"] = config["max_order"]
    if "index" in named:
        candidate["index"] = config["index"]
    candidate.update(overrides)

    if not has_var_kw:
        candidate = {k: v for k, v in candidate.items() if k in named}
    return cls(**candidate)


def _class_accepts_index(cls: type) -> bool:
    named, _ = _signature_params(cls)
    return "index" in named


# ===================================================================
# Protocol tests — every approximator must pass these
# ===================================================================


@pytest.mark.parametrize("config", ALL_APPROXIMATORS, ids=_approx_id)
class TestApproximatorProtocol:
    """Universal contract checks for all approximators."""

    def test_returns_interaction_values(self, config):
        """Approximate returns InteractionValues with correct metadata."""
        game = DummyGame(n=config["n"], interaction=(1, 2))
        approx = _make_approximator(config, random_state=42)
        result = approx.approximate(config["budget"], game)

        assert isinstance(result, InteractionValues)
        assert result.max_order == config["max_order"]
        assert result.n_players == config["n"]

    def test_respects_budget(self, config):
        """Game is not called more than budget + 2 times."""
        game = DummyGame(n=config["n"], interaction=(1, 2))
        approx = _make_approximator(config, random_state=42)
        approx.approximate(config["budget"], game)

        assert game.access_counter <= config["budget"] + 2

    def test_reproducible(self, config):
        """Same random_state produces identical results.

        Compare by ``interaction_lookup`` alignment rather than raw array
        ordering: SPEX's sparse transform produces the same interaction values
        but can place them in a different position in the ``values`` array
        across runs on some platforms (notably Windows). A small absolute
        tolerance is allowed for floating-point noise in post-processing.
        """
        game1 = DummyGame(n=config["n"], interaction=(1, 2))
        game2 = DummyGame(n=config["n"], interaction=(1, 2))
        a1 = _make_approximator(config, random_state=42)
        a2 = _make_approximator(config, random_state=42)
        r1 = a1.approximate(config["budget"], game1)
        r2 = a2.approximate(config["budget"], game2)

        assert_iv_close(r1, r2, atol=1e-3)

    def test_rejects_invalid_index(self, config):
        """Raises error for indices not in valid_indices."""
        if not _class_accepts_index(config["cls"]):
            pytest.skip(f"{config['cls'].__name__} does not accept an 'index' parameter")
        with pytest.raises((ValueError, TypeError)):
            _make_approximator(config, index="NOT_A_REAL_INDEX")


# ===================================================================
# Special cases
# ===================================================================


class TestProxySHAP:
    """ProxySHAP delegates to different adjustment strategies."""

    def test_default_adjustment_is_msr_biased(self):
        proxy = ProxySHAP(n=7, max_order=2, index="SII")
        assert proxy.adjustment == "msr-b"

    def test_approximate_runs(self):
        game = DummyGame(n=7, interaction=(1, 2))
        proxy = ProxySHAP(n=7, max_order=2, index="SII", random_state=42)
        result = proxy.approximate(100, game)
        assert isinstance(result, InteractionValues)
