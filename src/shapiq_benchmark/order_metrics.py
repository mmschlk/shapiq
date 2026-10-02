"""Order-specific errors; weak interaction truth is undefined, never a perfect score."""

from __future__ import annotations

import math

MIN_ORDER_SIGNAL_RATIO = 1e-6


def order_scores(truth: dict, prediction: dict, game: dict) -> dict:
    """Score each degree against payoff variation, falling back to full-target RMS.

    The fallback supports legacy/analytic games without a recorded payoff standard
    deviation. Its reference is explicit so these two signal tests are not confused.
    Sparse missing predictions are zero; the empty coefficient is never scored.
    """
    n, order = game["n_players"], game["order"]
    energy = math.fsum(value**2 for key, value in truth.items() if key)
    payoff_std = game.get("metadata", {}).get("payoff_std")
    reference = "payoff_std" if payoff_std is not None else "full_target_rms"
    scale = (
        payoff_std
        if payoff_std is not None
        else math.sqrt(energy / sum(math.comb(n, degree) for degree in range(1, order + 1)))
    )
    if not math.isfinite(scale) or scale < 0:
        message = "Invalid order-score signal reference."
        raise ValueError(message)
    scores = {}
    for degree in range(1, order + 1):
        coordinates = {key for key in truth.keys() | prediction.keys() if len(key) == degree}
        error = math.fsum(
            (prediction.get(key, 0.0) - truth.get(key, 0.0)) ** 2 for key in coordinates
        )
        degree_energy = math.fsum(value**2 for key, value in truth.items() if len(key) == degree)
        count = math.comb(n, degree)
        ratio = math.sqrt(degree_energy / count) / scale if scale else 0.0
        eligible = degree_energy > 0 and ratio >= MIN_ORDER_SIGNAL_RATIO
        if not all(math.isfinite(value) for value in (error, degree_energy, ratio)):
            message = "Nonfinite order-specific score."
            raise ValueError(message)
        scores[str(degree)] = {
            "nmse": error / degree_energy if eligible else None,
            "mse": error / count,
            "truth_energy": degree_energy,
            "energy_share": degree_energy / energy if energy else None,
            "signal_ratio": ratio,
            "signal_reference": reference,
            "score_eligible": eligible,
        }
    return scores
