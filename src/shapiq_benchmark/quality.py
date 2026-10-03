"""Versioned, score-independent checks on benchmark games and predictors.

These checks never inspect estimator errors. Controls retain their original
payoffs and exact answers; they are labeled separately from the primary panel.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shapiq.imputer.base import Imputer

QUALITY_PROTOCOL = "quality-v2"
QUALITY_POLICY: dict = {
    "version": QUALITY_PROTOCOL,
    "inactive_relative_tolerance": 1e-12,
    "empty_jump_control_threshold": 0.7,
    "maximum_imputation_noise_ratio": 0.1,
    "imputation_probe_coalitions": 32,
    "imputation_sample_multiplier": 4,
}


class QualityExclusion(ValueError):  # noqa: N818 -- a preflight decision, not an estimator error
    """An explicit, serializable preflight exclusion, not an estimator failure."""

    def __init__(self, reason: str, details: dict) -> None:
        """Retain a stable public reason and its measured evidence."""
        self.reason = reason
        self.details = details
        super().__init__(reason)


def validate_protocol(protocol: str | None) -> None:
    """Omission preserves legacy behavior; new policies must be named explicitly."""
    if protocol not in (None, QUALITY_PROTOCOL):
        message = f"Unknown game quality protocol: {protocol}"
        raise ValueError(message)


def model_validation_check(metadata: dict) -> dict:
    """Require improvement over a fitting-row dummy on validation only."""
    scores = metadata["quality"]["validation"]
    metric = "log_loss" if "log_loss" in scores["model"] else "mse"
    model, dummy = float(scores["model"][metric]), float(scores["dummy"][metric])
    return {
        "policy": "strictly lower validation loss than fitting-row dummy; test unused",
        "metric": metric,
        "model_loss": model,
        "dummy_loss": dummy,
        "passed": bool(np.isfinite(model) and np.isfinite(dummy) and model < dummy),
    }


def payoff_diagnostics(values: np.ndarray, n_players: int) -> dict:
    """Describe the entire frozen table, including empty-set and null-player artifacts."""
    values = np.asarray(values, dtype=float)
    if values.shape != (2**n_players,) or not np.isfinite(values).all():
        message = "Game diagnostics require a finite, complete canonical payoff table."
        raise ValueError(message)
    span = float(np.ptp(values))
    tolerance = QUALITY_POLICY["inactive_relative_tolerance"] * span
    # Reshaping pairs each coalition without i with the same coalition plus i.
    effects = [
        float(np.max(np.abs(block[:, width:] - block[:, :width])))
        for width in (2**i for i in range(n_players))
        for block in [values.reshape(-1, 2 * width)]
    ]
    inactive = [i for i, effect in enumerate(effects) if effect <= tolerance]
    variance = float(np.var(values, dtype=np.longdouble))
    empty_fraction = (
        float(
            (values[0] - np.mean(values[1:], dtype=np.longdouble)) ** 2
            * (len(values) - 1)
            / len(values) ** 2
            / variance
        )
        if variance
        else None
    )
    nonempty_constant = bool(np.ptp(values[1:]) <= tolerance)
    reasons = []
    if span == 0:
        reasons.append("constant_game")
    elif nonempty_constant:
        reasons.append("constant_nonempty_payoffs")
    if (
        empty_fraction is not None
        and empty_fraction > QUALITY_POLICY["empty_jump_control_threshold"]
    ):
        reasons.append("empty_coalition_jump")
    if inactive:
        reasons.append("inactive_players")
    return {
        "protocol": QUALITY_PROTOCOL,
        "payoff_range": span,
        "nonempty_constant": nonempty_constant,
        "empty_indicator_variance_fraction": empty_fraction,
        "empty_indicator_definition": "Var(E[v | empty vs nonempty]) / Var(v), uniform coalitions",
        "inactive_players": inactive,
        "active_players": n_players - len(inactive),
        "inactive_relative_tolerance": QUALITY_POLICY["inactive_relative_tolerance"],
        "role": "control" if reasons else "core",
        "control_reasons": reasons,
    }


def imputation_stability(game: Imputer, *, seed: int = 0) -> dict:
    """Probe Monte Carlo noise without refitting or changing the actual game.

    Copies share the fitted predictor and conditional distribution. Only the
    imputation RNG and sample count change. This small pilot estimates noise;
    it is not a confidence interval or a population ground-truth certificate.
    """
    masks = (
        np.random.default_rng(0)
        .integers(0, 2, size=(QUALITY_POLICY["imputation_probe_coalitions"], game.n_players))
        .astype(bool)
    )
    levels = []
    means = []
    for multiplier in (1, QUALITY_POLICY["imputation_sample_multiplier"]):
        repetitions = []
        for repeat in range(2):
            probe = copy.copy(game)
            probe.set_random_state(seed + 100_003 + repeat)
            probe._sample_size = game.sample_size * multiplier  # noqa: SLF001 -- isolated probe only
            repetitions.append(np.asarray(probe(masks), dtype=float))
        values = np.asarray(repetitions)
        if values.shape != (2, len(masks)) or not np.isfinite(values).all():
            reason = "invalid_imputation_pilot"
            raise QualityExclusion(reason, {"sample_multiplier": multiplier})
        noise = float(np.mean((values[0] - values[1]) ** 2) / 2)
        variation = float(np.var(values))
        means.append(values.mean(axis=0))
        levels.append(
            {
                "requested_sample_size": game.sample_size * multiplier,
                "noise_variance": noise,
                "pooled_variance": variation,
                "noise_ratio": noise / variation if variation else None,
            }
        )
    scale = max(level["pooled_variance"] for level in levels)
    drift = float(np.mean((means[0] - means[1]) ** 2)) / scale if scale else None
    limit = QUALITY_POLICY["maximum_imputation_noise_ratio"]
    stable = all(
        level["noise_ratio"] is not None and level["noise_ratio"] <= limit for level in levels
    )
    stable = stable and drift is not None and drift <= limit
    return {
        "protocol": QUALITY_PROTOCOL,
        "scope": "fixed predictor/input/background; imputation RNG and requested sample size only",
        "coalitions": len(masks),
        "repeats_per_level": 2,
        "levels": levels,
        "sample_size_drift_ratio": drift,
        "maximum_noise_ratio": limit,
        "status": "stable" if stable else "indeterminate" if not scale else "noise_dominated",
        "limitation": "Small uniform-coalition pilot; frozen-table exactness does not imply population accuracy.",
    }
