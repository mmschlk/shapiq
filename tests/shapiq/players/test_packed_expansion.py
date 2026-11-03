"""Direct packed expansion and portable logical-storage dispatch."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.players import FinitePlayers, finite


@pytest.mark.parametrize("id_itemsize", [1, 2, 4, 8])
def test_dispatch_compares_logical_bytes_at_the_exact_crossover(id_itemsize):
    n_players = 8
    index_size = jax.dtypes.canonicalize_dtype(int).itemsize
    # One set: four ID buffers, ten byte/flag buffers, three index vectors,
    # byte offsets, and one ID-valued scan seed. The shared output is excluded.
    packed_bytes = (
        n_players * (4 * id_itemsize + 10) + n_players * (3 * index_size + 1) + id_itemsize
    )
    latent_at_tie = packed_bytes - n_players
    assert not finite._use_sparse_packed_expansion(n_players, latent_at_tie, (), id_itemsize)
    assert finite._use_sparse_packed_expansion(n_players, latent_at_tie + 1, (), id_itemsize)


def test_dispatch_accounts_for_shared_indices_batch_size_and_id_width():
    # Shared position vectors are amortized across the batch; wider IDs cost
    # more prefix-scan storage. These outcomes follow byte counts, not timings.
    assert not finite._use_sparse_packed_expansion(8, 256, (), 4)
    assert finite._use_sparse_packed_expansion(8, 256, (16,), 4)
    assert not finite._use_sparse_packed_expansion(8, 256, (16,), 8)
    assert not finite._use_sparse_packed_expansion(64, 64, (64,), 4)
    assert finite._use_sparse_packed_expansion(8, 65536, (), 4)


def test_empty_expansions_use_direct_zero_filling():
    assert finite._use_sparse_packed_expansion(0, 0, (), 4)
    assert finite._use_sparse_packed_expansion(0, 32, (3,), 4)
    assert finite._use_sparse_packed_expansion(3, 0, (0,), 4)


@pytest.mark.parametrize("n_players", [0, 1, 7, 8, 9, 17, 64])
@pytest.mark.parametrize(
    ("player_batch", "coalition_batch"), [((), (3,)), ((2, 1), (3,)), ((0, 1), (3,))]
)
def test_direct_packed_expansion_matches_boolean_expansion(
    n_players, player_batch, coalition_batch
):
    latent_count = 3 * n_players + 13
    ids = jnp.broadcast_to(3 * jnp.arange(n_players) + 1, (*player_batch, n_players))
    players = FinitePlayers(ids, latent_n_players=latent_count)
    coalition = (
        jnp.arange(prod(coalition_batch) * n_players).reshape((*coalition_batch, n_players)) % 3
        == 0
    )
    packed = pack_bool_coalition(coalition)
    expected = pack_bool_coalition(players.expand_coalition(coalition))
    result = finite._expand_packed_coalition(players.ids, packed, latent_count)
    assert result.dtype == jnp.uint8
    assert result.shape == expected.shape
    assert bool(jnp.array_equal(result, expected))


def test_all_selections_of_duplicate_runs_are_carry_safe():
    ids = jnp.array([0, 0, 0, 1, 1, 7, 7, 7, 8, 8, 15])
    players = FinitePlayers(ids, latent_n_players=20)
    # Include every selection, especially True/False/True within one ID run.
    coalition = (
        (jnp.arange(1 << players.n_players)[:, None] >> jnp.arange(players.n_players)) & 1
    ).astype(bool)
    expected = pack_bool_coalition(players.expand_coalition(coalition))
    result = finite._expand_packed_coalition(players.ids, pack_bool_coalition(coalition), 20)
    assert bool(jnp.array_equal(result, expected))


def test_eight_selected_bits_in_one_byte_do_not_overflow():
    players = FinitePlayers(jnp.arange(16), latent_n_players=17)
    result = finite._expand_packed_coalition(
        players.ids, jnp.array([255, 255], dtype=jnp.uint8), 17
    )
    assert result.tolist() == [255, 255, 0]


@pytest.mark.parametrize(
    ("dtype", "last_id"), [(jnp.int8, 127), (jnp.int16, 32767), (jnp.int32, 65536)]
)
def test_direct_bit_positions_work_across_id_dtypes_and_byte_boundaries(dtype, last_id):
    players = FinitePlayers(
        jnp.array([0, 1, 7, 8, last_id], dtype=dtype), latent_n_players=last_id + 1
    )
    coalition = jnp.array([True, False, True, True, True])
    packed = pack_bool_coalition(coalition) | jnp.uint8(0b11100000)
    expected = pack_bool_coalition(players.expand_coalition(coalition))
    result = finite._expand_packed_coalition(players.ids, packed, players.latent_n_players)
    assert bool(jnp.array_equal(result, expected))


def test_sparse_expansion_does_not_normalize_narrow_ids_against_the_latent_size():
    players = FinitePlayers(jnp.array([0, 127], dtype=jnp.int8), latent_n_players=65537)
    result = players.expand_coalition(jnp.array([3], dtype=jnp.uint8))
    expected = np.zeros((65537 + 7) // 8, dtype=np.uint8)
    expected[0] = 1
    expected[15] = 128
    assert np.array_equal(result, expected)


def test_sparse_expansion_supports_int64_ids_and_index_vectors():
    with jax.enable_x64(new_val=True):
        players = FinitePlayers(
            jnp.array([0, 127, 65536], dtype=jnp.int64), latent_n_players=131073
        )
        result = players.expand_coalition(jnp.array([5], dtype=jnp.uint8))
        expected = np.zeros((131073 + 7) // 8, dtype=np.uint8)
        expected[0] = expected[8192] = 1
        assert np.array_equal(result, expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_sparse_public_path_does_not_unpack_or_allocate_boolean_latent_membership(
    monkeypatch, compiled
):
    players = FinitePlayers(
        jnp.array(
            [0, 1, 2, 7, 8, 15, 16, 255, 256, 257, 1023, 4095, 4096, 32000, 32001, 65534, 65536]
        ),
        latent_n_players=65537,
    )
    coalition = jnp.stack(
        [jnp.arange(players.n_players) % 2 == 0, jnp.arange(players.n_players) % 3 == 0]
    )
    expected = pack_bool_coalition(players.expand_coalition(coalition))
    packed = pack_bool_coalition(coalition)
    packed = packed.at[..., -1].set(packed[..., -1] | jnp.uint8(0b11111110))

    def unexpected_boolean_path(*args, **kwargs):
        msg = "Sparse packed expansion must not use boolean conversion or latent membership"
        raise AssertionError(msg)

    monkeypatch.setattr(finite, "unpack_bool_coalition", unexpected_boolean_path)
    monkeypatch.setattr(finite, "pack_bool_coalition", unexpected_boolean_path)
    monkeypatch.setattr(finite, "_expand_bool_coalition", unexpected_boolean_path)

    def expand(p, c):
        return p.expand_coalition(c)

    if compiled:
        expand = jax.jit(expand)
    result = expand(players, packed)
    assert np.array_equal(result, expected)
    assert not bool(jnp.any(result[..., -1] & jnp.uint8(0b11111110)))


@pytest.mark.parametrize(("n_players", "latent_count"), [(8, 64), (64, 64)])
def test_dense_public_path_keeps_boolean_expansion(monkeypatch, n_players, latent_count):
    players = FinitePlayers(n_players, latent_n_players=latent_count)
    coalition = jnp.arange(n_players) % 2 == 0
    expected = pack_bool_coalition(players.expand_coalition(coalition))

    def unexpected_packed_path(*args, **kwargs):
        msg = "Dense expansion must not use the more expensive packed bookkeeping"
        raise AssertionError(msg)

    monkeypatch.setattr(finite, "_expand_packed_coalition", unexpected_packed_path)
    result = players.expand_coalition(pack_bool_coalition(coalition))
    assert bool(jnp.array_equal(result, expected))


@pytest.mark.parametrize("packed_expansion", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("latent_count", [20, 65537])
def test_explicit_packed_expansion_bypasses_auto_dispatch(
    monkeypatch, packed_expansion, compiled, latent_count
):
    ids = jnp.broadcast_to(jnp.array([0, 0, 1, 3, 7, 8, 8, 15, 16]), (2, 1, 9))
    players = FinitePlayers(ids, latent_n_players=latent_count, sorted=True)
    coalition = jnp.arange(27).reshape((3, 9)) % 3 != 0
    expected = pack_bool_coalition(players.expand_coalition(coalition))
    packed = pack_bool_coalition(coalition)
    packed = packed.at[..., -1].set(packed[..., -1] | jnp.uint8(0b11111110))

    def unexpected_path(*args, **kwargs):
        msg = "An explicit packed_expansion must bypass auto dispatch and the other path"
        raise AssertionError(msg)

    monkeypatch.setattr(finite, "_use_sparse_packed_expansion", unexpected_path)
    if packed_expansion:
        monkeypatch.setattr(finite, "_expand_bool_coalition", unexpected_path)
        monkeypatch.setattr(finite, "unpack_bool_coalition", unexpected_path)
        monkeypatch.setattr(finite, "pack_bool_coalition", unexpected_path)
    else:
        monkeypatch.setattr(finite, "_expand_packed_coalition", unexpected_path)

    def expand(p, c, *, packed_expansion):
        return p.expand_coalition(c, packed_expansion=packed_expansion)

    if compiled:
        expand = jax.jit(expand, static_argnames=("packed_expansion",))
    result = expand(players, packed, packed_expansion=packed_expansion)
    assert result.dtype == jnp.uint8
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("packed_expansion", [False, "auto"])
@pytest.mark.parametrize("compiled", [False, True])
def test_boolean_input_keeps_boolean_expansion(monkeypatch, packed_expansion, compiled):
    players = FinitePlayers(jnp.array([1, 4]), latent_n_players=7)
    coalition = jnp.array([True, False])

    def unexpected_path(*args, **kwargs):
        msg = "Boolean input must not use packed expansion or auto dispatch"
        raise AssertionError(msg)

    monkeypatch.setattr(finite, "_use_sparse_packed_expansion", unexpected_path)
    monkeypatch.setattr(finite, "_expand_packed_coalition", unexpected_path)

    def expand(p, c):
        return p.expand_coalition(c, packed_expansion=packed_expansion)

    if compiled:
        expand = jax.jit(expand)
    result = expand(players, coalition)
    assert result.dtype == jnp.bool_
    assert result.tolist() == [False, True, False, False, False, False, False]


@pytest.mark.parametrize("n_players", [0, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_forcing_packed_expansion_rejects_boolean_input(n_players, compiled):
    players = FinitePlayers(n_players, latent_n_players=7)
    coalition = jnp.zeros(n_players, dtype=bool)

    def expand(p, c):
        return p.expand_coalition(c, packed_expansion=True)

    if compiled:
        expand = jax.jit(expand)
    with pytest.raises(ValueError, match="packed_expansion=True requires a packed uint8"):
        expand(players, coalition)


@pytest.mark.parametrize("packed_expansion", [False, True, "auto"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("player_batch", "coalition_batch", "n_players", "latent_count"),
    [((), (), 0, 0), ((), (3,), 0, 5), ((0,), (), 2, 0), ((0, 1), (3,), 0, 7)],
)
def test_packed_expansion_options_handle_empty_sets_and_batches(
    packed_expansion, compiled, *, player_batch, coalition_batch, n_players, latent_count
):
    players = FinitePlayers(
        jnp.zeros((*player_batch, n_players), dtype=int), latent_n_players=latent_count
    )
    coalition = pack_bool_coalition(jnp.zeros((*coalition_batch, n_players), dtype=bool))

    def expand(p, c):
        return p.expand_coalition(c, packed_expansion=packed_expansion)

    if compiled:
        expand = jax.jit(expand)
    result = expand(players, coalition)
    batch = jnp.broadcast_shapes(player_batch, coalition_batch)
    assert result.dtype == jnp.uint8
    assert result.shape == (*batch, (latent_count + 7) // 8)
    assert not bool(jnp.any(result))
