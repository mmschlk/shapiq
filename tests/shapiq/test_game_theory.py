"""Tests for game theory module: ExactComputer, indices, Moebius converter, core, aggregation, Game."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.game import Game
from shapiq.game_theory.aggregation import (
    aggregate_base_interaction,
    aggregate_to_one_dimension,
)
from shapiq.game_theory.core import egalitarian_least_core
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


# ===================================================================
# Aggregation
# ===================================================================


class TestAggregation:
    """Tests for game_theory.aggregation."""

    def test_aggregate_base_interaction_sii_to_k_sii(self):
        """SII -> k-SII aggregation renames the index and preserves structure."""
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        sii = computer("SII", order=2)

        k_sii = aggregate_base_interaction(sii)

        assert isinstance(k_sii, InteractionValues)
        assert k_sii.index == "k-SII"
        assert k_sii.n_players == sii.n_players
        assert k_sii.max_order == sii.max_order

        # k-SII is efficient.
        grand = float(game(np.ones((1, 3), dtype=bool))[0])
        empty = float(game(np.zeros((1, 3), dtype=bool))[0])
        assert float(np.sum(k_sii.values)) == pytest.approx(
            grand - empty - k_sii.baseline_value, abs=1e-10
        )

    def test_aggregate_to_one_dimension_shapes(self):
        """aggregate_to_one_dimension returns (pos, neg) vectors of length n_players."""
        game = DummyGame(n=4, interaction=(0, 1))
        computer = ExactComputer(game)
        k_sii = computer("k-SII", order=2)

        pos, neg = aggregate_to_one_dimension(k_sii)
        assert pos.shape == (4,)
        assert neg.shape == (4,)
        # positive values are non-negative and negative values are non-positive
        assert (pos >= 0).all()
        assert (neg <= 0).all()


# ===================================================================
# Core
# ===================================================================


class TestCore:
    """Tests for game_theory.core.egalitarian_least_core."""

    def test_elc_shape_and_type(self):
        """Returns an InteractionValues of order 1 plus a non-negative subsidy."""
        from shapiq.utils import powerset

        n = 3
        game = DummyGame(n=n, interaction=(0, 1))
        coalitions = list(powerset(range(n)))
        values = np.array(
            [float(game(np.array([[i in c for i in range(n)]], dtype=bool))[0]) for c in coalitions]
        )
        # Pre-shift so the empty set value is 0 (egalitarian_least_core warns otherwise).
        values = values - values[0]
        lookup_aligned = {coal: i for i, coal in enumerate(coalitions)}

        elc, subsidy = egalitarian_least_core(
            n_players=n, game_values=values, coalition_lookup=lookup_aligned
        )
        assert isinstance(elc, InteractionValues)
        assert elc.index == "ELC"
        assert elc.n_players == n
        assert elc.max_order == 1
        assert subsidy >= 0

        # Efficiency: sum of credits equals the grand coalition value.
        grand_value = values[lookup_aligned[tuple(range(n))]]
        assert float(np.sum(elc.values)) == pytest.approx(grand_value, abs=1e-4)


# ===================================================================
# Game base class
# ===================================================================


class TestGame:
    """Tests for the :class:`shapiq.game.Game` base class via ``DummyGame``."""

    def test_call_returns_correct_shape(self):
        game = DummyGame(n=4, interaction=(0, 1))
        coalitions = np.eye(4, dtype=bool)
        values = game(coalitions)
        assert values.shape == (4,)

    def test_access_counter_increments(self):
        game = DummyGame(n=3, interaction=(0, 1))
        assert game.access_counter == 0
        game(np.ones((2, 3), dtype=bool))
        assert game.access_counter == 2

    def test_grand_and_empty_coalition_values(self):
        game = DummyGame(n=3, interaction=(0, 1))
        grand = float(game(np.ones((1, 3), dtype=bool))[0])
        empty = float(game(np.zeros((1, 3), dtype=bool))[0])
        assert game.grand_coalition_value == pytest.approx(grand)
        assert game.empty_coalition_value == pytest.approx(empty)

    def test_precompute_sets_flag_and_stores_values(self):
        game = DummyGame(n=3, interaction=(0, 1))
        assert game.precomputed is False
        game.precompute()
        assert game.precomputed is True
        assert game.n_values_stored == 2**3

    def test_save_and_load_values_roundtrip(self, tmp_path):
        game = DummyGame(n=3, interaction=(0, 1))
        game.precompute()
        path = tmp_path / "vals.npz"
        game.save_values(path, as_npz=True)

        loaded = DummyGame(n=3, interaction=(0, 1))
        loaded.load_values(path, precomputed=True)
        assert loaded.precomputed is True
        # values lookup should match for the grand coalition
        coals = np.ones((1, 3), dtype=bool)
        assert loaded(coals)[0] == pytest.approx(game(coals)[0])

    def test_save_and_load_game_json(self, tmp_path):
        """Game.save/load performs a full-object round-trip via JSON."""
        game = DummyGame(n=3, interaction=(0, 1))
        game.precompute()
        path = tmp_path / "game.json"
        game.save(path)

        loaded = Game.load(path)
        assert loaded.n_players == 3
        assert loaded.precomputed is True
