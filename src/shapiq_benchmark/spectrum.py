"""Describe a frozen game using its orthogonal Boolean Fourier spectrum.

Coalitions have uniform weight here. This is a property of the payoff table, not
an interaction index, estimator score, or population spectrum for a noisy game.
"""

from __future__ import annotations

import numpy as np


def fourier_spectrum(values: np.ndarray, n_players: int) -> dict:
    """Return nonconstant energy by order using a stable Walsh-Hadamard transform.

    Centering and scaling remove the arbitrary payoff offset/scale before the
    transform. The work is O(d * 2**d), with no additional game evaluations.
    Tiny numerical tails are reported rather than used as selection thresholds.
    """
    values = np.asarray(values, dtype=np.longdouble)
    if values.shape != (2**n_players,) or not np.isfinite(values).all():
        message = "The Fourier spectrum requires a finite canonical coalition table."
        raise ValueError(message)
    coefficients = values - np.mean(values)
    scale = np.max(np.abs(coefficients))
    result = {
        "measure": "uniform_coalition_boolean_fourier",
        "normalization": "fraction of nonconstant payoff variance",
        "degree_mass": [0.0] * (n_players + 1),
        "mass_above_three": 0.0,
        "effective_order_90": 0,
        "constant": bool(scale == 0),
    }
    if scale == 0:
        return result
    coefficients /= scale
    width = 1
    while width < len(coefficients):
        blocks = coefficients.reshape(-1, 2 * width)
        left, right = blocks[:, :width].copy(), blocks[:, width:].copy()
        blocks[:, :width], blocks[:, width:] = left + right, left - right
        width *= 2
    coefficients /= len(coefficients)
    energy = np.asarray(coefficients**2, dtype=float)
    energy[0] = 0  # The constant payoff offset does not describe interaction structure.
    degrees = np.fromiter((mask.bit_count() for mask in range(len(values))), dtype=int)
    masses = np.bincount(degrees, weights=energy, minlength=n_players + 1)
    masses /= masses.sum()
    result.update(
        degree_mass=masses.tolist(),
        mass_above_three=float(masses[4:].sum()),
        effective_order_90=int(np.searchsorted(np.cumsum(masses), 0.9)),
    )
    return result
