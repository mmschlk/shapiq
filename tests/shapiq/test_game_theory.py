"""Tests for game theory module: ExactComputer, indices, Moebius converter."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.game_theory.exact import ExactComputer
from shapiq.game_theory.indices import (
    get_computation_index,
    index_generalizes_bv,
    index_generalizes_sv,
    is_index_valid,
)
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame


# ===================================================================
# ExactComputer
# ===================================================================


class TestExactComputer:
    """Tests for exact computation of interaction indices."""

    def test_sv_on_dummy_game(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        sv = computer("SV", order=1)

        assert isinstance(sv, InteractionValues)
        assert sv.index == "SV"
        assert sv.n_players == 3

        # SV satisfies efficiency: sum = game(N) - game(empty)
        grand = float(game(np.ones((1, 3), dtype=bool))[0])
        empty = float(game(np.zeros((1, 3), dtype=bool))[0])
        assert float(np.sum(sv.values)) == pytest.approx(grand - empty, abs=1e-10)

    def test_sii_on_dummy_game(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        sii = computer("SII", order=2)

        assert isinstance(sii, InteractionValues)
        assert sii.index == "SII"
        assert sii.max_order == 2

    def test_k_sii_on_dummy_game(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        k_sii = computer("k-SII", order=2)

        assert isinstance(k_sii, InteractionValues)
        assert k_sii.index == "k-SII"

        # k-SII satisfies efficiency
        grand = float(game(np.ones((1, 3), dtype=bool))[0])
        empty = float(game(np.zeros((1, 3), dtype=bool))[0])
        assert float(np.sum(k_sii.values)) == pytest.approx(grand - empty, abs=1e-10)

    @pytest.mark.parametrize("index", ["SV", "SII", "k-SII", "STII", "FSII", "FBII", "BV", "BII"])
    def test_all_common_indices(self, index):
        """ExactComputer should handle all common indices without error."""
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        order = 1 if index in ("SV", "BV") else 2
        result = computer(index, order=order)
        assert isinstance(result, InteractionValues)

    def test_moebius_values(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        moebius = computer("Moebius", order=3)
        assert isinstance(moebius, InteractionValues)


# ===================================================================
# Index utilities
# ===================================================================


class TestIndices:
    def test_is_index_valid_true(self):
        assert is_index_valid("SV")
        assert is_index_valid("k-SII")

    def test_is_index_valid_false(self):
        assert not is_index_valid("NOT_REAL")

    def test_is_index_valid_raises(self):
        with pytest.raises(ValueError):
            is_index_valid("NOT_REAL", raise_error=True)

    def test_generalizes_sv(self):
        assert index_generalizes_sv("SII")
        assert index_generalizes_sv("k-SII")
        assert not index_generalizes_sv("BV")
        assert not index_generalizes_sv("SV")

    def test_generalizes_bv(self):
        assert index_generalizes_bv("BII")
        assert not index_generalizes_bv("SII")

    def test_get_computation_index(self):
        assert get_computation_index("k-SII") == "SII"
        assert get_computation_index("SV") == "SII"
        assert get_computation_index("BV") == "BII"
        assert get_computation_index("STII") == "STII"


# ===================================================================
# Moebius converter
# ===================================================================


class TestMoebiusConverter:
    def test_sii_to_k_sii_roundtrip(self):
        """Convert SII -> k-SII and verify it's a valid InteractionValues."""
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        sii = computer("SII", order=2)

        converter = MoebiusConverter(sii)
        k_sii = converter("k-SII")

        assert isinstance(k_sii, InteractionValues)
        assert k_sii.index == "k-SII"
