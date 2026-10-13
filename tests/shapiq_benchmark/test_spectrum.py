"""Boolean spectral order must reflect known games and ignore payoff offset/scale."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_benchmark.spectrum import fourier_spectrum


def test_known_orthogonal_components_and_affine_invariance() -> None:
    """Separate additive and fourth-order variation even when a constant dominates."""
    masks = np.arange(64)
    singleton = 1 - 2 * (masks & 1)
    parity = np.array([(-1) ** (int(mask) & 15).bit_count() for mask in masks])
    values = 2 * singleton + 3 * parity
    expected = np.zeros(7)
    expected[1], expected[4] = 4 / 13, 9 / 13
    for table in (values, values + 10**8, values * 1e-200):
        spectrum = fourier_spectrum(table, 6)
        np.testing.assert_allclose(spectrum["degree_mass"], expected, atol=1e-15)
        assert spectrum["mass_above_three"] == pytest.approx(9 / 13)
        assert spectrum["effective_order_90"] == 4
        assert not spectrum["constant"]


def test_constant_game_has_no_interaction_energy() -> None:
    spectrum = fourier_spectrum(np.full(32, 7.0), 5)
    assert spectrum["constant"]
    assert spectrum["degree_mass"] == [0] * 6
    assert spectrum["effective_order_90"] == 0


@pytest.mark.parametrize("values", [np.ones(7), np.array([0, 1, 2, np.nan])])
def test_invalid_tables_are_rejected(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite canonical"):
        fourier_spectrum(values, 2)
