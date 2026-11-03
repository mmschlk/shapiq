"""Latent-to-local coalition restriction for boolean and packed membership."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from shapiq.coalition.coalition import pack_bool_coalition, unpack_bool_coalition
from shapiq.players import FinitePlayers, finite


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_contraction_maps_latent_membership_to_sorted_local_positions(packed, compiled):
    players = FinitePlayers(jnp.array([4, 1]), latent_n_players=7)
    coalition = jnp.array([True, True, True, False, False, False, True])
    if packed:
        coalition = pack_bool_coalition(coalition)

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    result = contract(players, coalition)
    assert result.dtype == (jnp.uint8 if packed else jnp.bool_)
    assert players.is_valid(result)
    if packed:
        result = unpack_bool_coalition(result, 2)
    assert result.tolist() == [True, False]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("player_batch", "coalition_batch", "expected_batch"),
    [
        ((), (3,), (3,)),
        ((3,), (), (3,)),
        ((1,), (3,), (3,)),
        ((2, 1), (3,), (2, 3)),
        ((3,), (2, 1), (2, 3)),
        ((2, 3), (1, 2, 3), (1, 2, 3)),
        ((1,), (0,), (0,)),
        ((0, 1), (3,), (0, 3)),
    ],
)
def test_contraction_broadcasts_symmetrically(
    packed, compiled, player_batch, coalition_batch, expected_batch
):
    ids = jnp.broadcast_to(jnp.array([1, 4]), (*player_batch, 2))
    players = FinitePlayers(ids, latent_n_players=7)
    membership = jnp.array([True, True, True, False, False, False, True])
    coalition = jnp.broadcast_to(membership, (*coalition_batch, 7))
    if packed:
        coalition = pack_bool_coalition(coalition)

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    result = contract(players, coalition)
    expected = np.broadcast_to(np.array([True, False]), (*expected_batch, 2))
    if packed:
        expected = np.packbits(expected, axis=-1, bitorder="little")
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_contraction_broadcasts_different_sets_and_selections_together(packed, compiled):
    players = FinitePlayers(jnp.array([[[1, 4]], [[0, 3]]]), latent_n_players=6)
    coalition = jnp.array(
        [
            [True, False, True, False, True, False],
            [False, True, True, True, False, True],
            [True, True, True, True, True, True],
        ]
    )
    expected = np.array(
        [[[False, True], [True, False], [True, True]], [[True, False], [False, True], [True, True]]]
    )
    if packed:
        coalition = pack_bool_coalition(coalition)
        expected = np.packbits(expected, axis=-1, bitorder="little")

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    assert np.array_equal(contract(players, coalition), expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_grand_latent_coalition_contracts_to_each_remaining_grand_coalition(packed, compiled):
    players = FinitePlayers(jnp.array([[0, 2, 5], [1, 3, 6]]), latent_n_players=8)
    players = players.remove(jnp.array([[2], [3]]))
    coalition = jnp.ones(8, dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    result = contract(players, coalition)
    if packed:
        result = unpack_bool_coalition(result, players.n_players)
    assert players.latent_n_players == 8
    assert result.tolist() == [[True, True], [True, True]]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("n_players", [0, 1, 7, 8, 9, 16, 17, 64])
def test_contraction_crosses_byte_boundaries(packed, compiled, n_players):
    ids = 2 * np.arange(n_players) + 1
    latent_count = 2 * n_players + 3
    players = FinitePlayers(jnp.asarray(ids), latent_n_players=latent_count)
    membership = np.arange(3 * latent_count).reshape((3, latent_count)) % 3 == 0
    expected = membership[..., ids]
    coalition = jnp.asarray(membership)
    if packed:
        coalition = pack_bool_coalition(coalition)
        expected = np.packbits(expected, axis=-1, bitorder="little")

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    assert np.array_equal(contract(players, coalition), expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_contraction_repeats_duplicate_ids_for_every_latent_selection(packed, compiled):
    ids = np.array([0, 0, 1, 7, 8, 8, 9])
    players = FinitePlayers(jnp.asarray(ids), latent_n_players=10)
    membership = ((np.arange(1 << 10)[:, None] >> np.arange(10)) & 1).astype(bool)
    expected = membership[:, ids]
    coalition = jnp.asarray(membership)
    if packed:
        coalition = pack_bool_coalition(coalition)
        expected = np.packbits(expected, axis=-1, bitorder="little")

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    assert np.array_equal(contract(players, coalition), expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_expansion_then_contraction_round_trips_unique_ids(packed, compiled):
    players = FinitePlayers(jnp.array([0, 1, 7, 8, 15, 16, 127, 128, 256]), latent_n_players=259)
    membership = jnp.arange(27).reshape((3, 9)) % 3 == 0
    coalition = pack_bool_coalition(membership) if packed else membership

    def round_trip(p, c):
        return p.contract_coalition(p.expand_coalition(c))

    if compiled:
        round_trip = jax.jit(round_trip)
    assert np.array_equal(round_trip(players, coalition), coalition)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_contraction_then_expansion_drops_unrepresented_latent_players(packed, compiled):
    ids = np.array([1, 3, 3, 8])
    players = FinitePlayers(jnp.asarray(ids), latent_n_players=11)
    membership = np.arange(33).reshape((3, 11)) % 2 == 0
    expected = np.zeros_like(membership)
    expected[:, ids] = membership[:, ids]
    coalition = jnp.asarray(membership)
    if packed:
        coalition = pack_bool_coalition(coalition)
        expected = np.packbits(expected, axis=-1, bitorder="little")

    def round_trip(p, c):
        return p.expand_coalition(p.contract_coalition(c))

    if compiled:
        round_trip = jax.jit(round_trip)
    assert np.array_equal(round_trip(players, coalition), expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_expansion_then_contraction_merges_duplicate_local_selections(packed, compiled):
    players = FinitePlayers(jnp.array([2, 2, 5]), latent_n_players=7)
    coalition = jnp.array([True, False, True])
    expected = np.array([True, True, True])
    if packed:
        coalition = pack_bool_coalition(coalition)
        expected = np.packbits(expected, bitorder="little")

    def round_trip(p, c):
        return p.contract_coalition(p.expand_coalition(c))

    if compiled:
        round_trip = jax.jit(round_trip)
    assert np.array_equal(round_trip(players, coalition), expected)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(("coalition", "expected"), [([0, 0b11100000], [0]), ([255, 255], [7])])
def test_packed_contraction_ignores_input_padding_and_zeros_output_padding(
    compiled, coalition, expected
):
    players = FinitePlayers(jnp.array([0, 4, 8]), latent_n_players=13)

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    result = contract(players, jnp.array(coalition, dtype=jnp.uint8))
    assert result.tolist() == expected


@pytest.mark.parametrize("compiled", [False, True])
def test_packed_contraction_never_unpacks_the_full_latent_coalition(monkeypatch, compiled):
    players = FinitePlayers(jnp.array([0, 1, 255, 256, 65536]), latent_n_players=65537)
    coalition = np.zeros((65537 + 7) // 8, dtype=np.uint8)
    coalition[0] = coalition[32] = 1
    coalition[-1] = 255  # Include dirty latent padding.

    def unexpected_unpack(*args, **kwargs):
        msg = "Packed contraction must only extract the referenced latent bits"
        raise AssertionError(msg)

    monkeypatch.setattr(finite, "unpack_bool_coalition", unexpected_unpack)
    monkeypatch.setattr(jnp, "unpackbits", unexpected_unpack)

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    result = contract(players, jnp.asarray(coalition))
    assert result.tolist() == [0b11001]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("dtype", "last_id", "latent_count"),
    [
        (jnp.int8, 127, 65537),
        (jnp.int16, 32767, 32769),
        (jnp.int32, 65536, 65537),
        (jnp.int64, 65536, 65537),
    ],
)
def test_contraction_supports_narrow_and_wide_ids(packed, compiled, dtype, last_id, latent_count):
    with jax.enable_x64(new_val=dtype == jnp.int64):
        ids = np.array([0, 1, 7, 8, last_id])
        players = FinitePlayers(jnp.array(ids, dtype=dtype), latent_n_players=latent_count)
        membership = np.arange(latent_count) % 3 == 1
        expected = membership[ids]
        coalition = jnp.asarray(membership)
        if packed:
            coalition = pack_bool_coalition(coalition)
            expected = np.packbits(expected, bitorder="little")

        def contract(p, c):
            return p.contract_coalition(c)

        if compiled:
            contract = jax.jit(contract)
        assert np.array_equal(contract(players, coalition), expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("player_batch", "coalition_batch", "n_players", "latent_count"),
    [
        ((), (), 0, 0),
        ((), (3,), 0, 5),
        ((0,), (), 2, 0),
        ((0, 1), (3,), 2, 0),
        ((2, 1), (3,), 0, 7),
    ],
)
def test_contraction_handles_empty_sets_and_batches(
    packed, compiled, *, player_batch, coalition_batch, n_players, latent_count
):
    players = FinitePlayers(
        jnp.zeros((*player_batch, n_players), dtype=int), latent_n_players=latent_count
    )
    coalition = jnp.zeros((*coalition_batch, latent_count), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)

    def contract(p, c):
        return p.contract_coalition(c)

    if compiled:
        contract = jax.jit(contract)
    result = contract(players, coalition)
    batch = jnp.broadcast_shapes(player_batch, coalition_batch)
    width = (n_players + 7) // 8 if packed else n_players
    assert result.shape == (*batch, width)
    assert result.dtype == (jnp.uint8 if packed else jnp.bool_)
    assert not bool(jnp.any(result))


@pytest.mark.parametrize(
    "coalition",
    [
        None,
        [True] * 9,
        np.ones(9, dtype=bool),
        jnp.ones(9),
        jnp.ones(9, dtype=jnp.int8),
        jnp.ones(8, dtype=bool),
        jnp.ones(1, dtype=jnp.uint8),
        jnp.ones((), dtype=bool),
    ],
)
def test_contraction_checks_latent_coalition_type_and_width(coalition):
    players = FinitePlayers(jnp.array([1, 3]), latent_n_players=9)
    with pytest.raises(TypeCheckError):
        players.contract_coalition(coalition)


@pytest.mark.parametrize("packed", [False, True])
def test_contraction_rejects_incompatible_batches(packed):
    players = FinitePlayers(jnp.array([[1, 3], [0, 2]]), latent_n_players=7)
    coalition = jnp.ones((3, 7), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    with pytest.raises(ValueError, match="Inadmissible latent coalition batch"):
        players.contract_coalition(coalition)
