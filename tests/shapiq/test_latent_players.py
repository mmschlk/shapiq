"""Shared latent player spaces and local-to-latent coalition expansion."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from shapiq.coalition.coalition import pack_bool_coalition, unpack_bool_coalition
from shapiq.players import FinitePlayers


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (jnp.array([2, 7]), 8),
        (jnp.array([[0, 2], [9, 1]]), 10),
        (jnp.empty((0,), dtype=int), 0),
        (jnp.empty((2, 0), dtype=int), 0),
        (jnp.empty((0, 3), dtype=int), 0),
    ],
)
def test_ids_infer_one_latent_count_for_the_whole_batch(ids, expected):
    players = FinitePlayers(ids)
    assert players.latent_n_players == expected
    assert isinstance(players.latent_n_players, int)


@pytest.mark.parametrize("keyword_ids", [False, True])
def test_explicit_latent_count_is_keyword_only_with_either_id_form(keyword_ids):
    ids = jnp.array([1, 4])
    if keyword_ids:
        players = FinitePlayers(ids=ids, latent_n_players=7)
    else:
        players = FinitePlayers(ids, latent_n_players=7)
    assert players.n_players == 2
    assert players.latent_n_players == 7
    assert players.ids.tolist() == [1, 4]
    assert players.players is None


def test_metadata_determines_latent_count_including_unreferenced_entries():
    metadata = list("abcdefghij")
    players = FinitePlayers(jnp.array([[0, 2], [1, 3]]), metadata)
    assert players.n_players == 2
    assert players.latent_n_players == len(metadata)
    assert list(players.players) == metadata


@pytest.mark.parametrize("metadata", [[], list("abc")])
@pytest.mark.parametrize("extra_count", [0, 1])
@pytest.mark.parametrize("form", ["players", "ids-players", "players-ids", "keywords", "mixed"])
def test_metadata_allows_a_consistent_explicit_latent_count(metadata, extra_count, form):
    ids = jnp.arange(len(metadata))
    count = len(metadata) + extra_count

    def construct():
        if form == "players":
            return FinitePlayers(metadata, latent_n_players=count)
        if form == "ids-players":
            return FinitePlayers(ids, metadata, latent_n_players=count)
        if form == "players-ids":
            return FinitePlayers(metadata, ids, latent_n_players=count)
        if form == "keywords":
            return FinitePlayers(ids=ids, players=metadata, latent_n_players=count)
        return FinitePlayers(ids, players=metadata, latent_n_players=count)

    if extra_count:
        with pytest.raises(ValueError, match=r"len\(players\)"):
            construct()
    else:
        players = construct()
        assert players.latent_n_players == len(metadata)
        assert list(players.players) == metadata


@pytest.mark.parametrize("n_players", [0, 3])
@pytest.mark.parametrize("extra_count", [0, 2])
@pytest.mark.parametrize("keyword_count", [False, True])
def test_local_count_accepts_a_matching_or_larger_latent_count(
    n_players, extra_count, keyword_count
):
    latent_count = n_players + extra_count
    players = (
        FinitePlayers(n_players=n_players, latent_n_players=latent_count)
        if keyword_count
        else FinitePlayers(n_players, latent_n_players=latent_count)
    )
    assert players.n_players == n_players
    assert players.latent_n_players == latent_count
    assert players.ids.tolist() == list(range(n_players))


@pytest.mark.parametrize(("n_players", "latent_count"), [(0, -1), (3, 2)])
def test_generated_ids_must_fit_the_explicit_latent_count(n_players, latent_count):
    with pytest.raises(ValueError, match="latent_n_players must be at least n_players"):
        FinitePlayers(n_players, latent_n_players=latent_count)


@pytest.mark.parametrize("metadata", [None, list("abcdef")])
def test_known_latent_counts_skip_min_max_inference_but_still_check_bounds(monkeypatch, metadata):
    ids = jnp.array([[0, 2], [1, 3]])
    out_of_bounds = jnp.array([0, 6])

    def unexpected_reduction(*args, **kwargs):
        msg = "A known latent count must not require min/max inference"
        raise AssertionError(msg)

    monkeypatch.setattr(jnp, "min", unexpected_reduction)
    monkeypatch.setattr(jnp, "max", unexpected_reduction)
    players = FinitePlayers(ids, players=metadata, latent_n_players=6, sorted=True)
    assert players.latent_n_players == 6
    with pytest.raises(ValueError, match=r"latent_n_players|metadata catalog"):
        FinitePlayers(out_of_bounds, players=metadata, latent_n_players=6, sorted=True)


@pytest.mark.parametrize("latent_count", [-1, 0, 4])
def test_explicit_latent_count_must_cover_every_id(latent_count):
    with pytest.raises(ValueError, match="latent_n_players"):
        FinitePlayers(jnp.array([1, 4]), latent_n_players=latent_count)


@pytest.mark.parametrize("ids", [[-1], [-1, 3], [[0, 1], [-1, 2]]])
def test_player_ids_are_nonnegative_latent_indices(ids):
    with pytest.raises(ValueError, match="0 <= ids"):
        FinitePlayers(jnp.array(ids))


@pytest.mark.parametrize("latent_count", [1.5, "5", jnp.array(5)])
def test_explicit_latent_count_is_a_python_integer(latent_count):
    with pytest.raises(TypeCheckError):
        FinitePlayers(jnp.array([1, 3]), latent_n_players=latent_count)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((jnp.array([1, 4]), 7), {}),
        ((7, jnp.array([1, 4])), {}),
        ((7,), {"ids": jnp.array([1, 4])}),
        ((jnp.array([1, 4]), 7), {"latent_n_players": 7}),
        ((7, jnp.array([1, 4])), {"latent_n_players": 7}),
    ],
)
def test_positional_integers_cannot_be_latent_counts_with_ids(args, kwargs):
    with pytest.raises(TypeError, match="n_players cannot be combined"):
        FinitePlayers(*args, **kwargs)


def test_latent_count_alone_does_not_specify_a_player_set():
    with pytest.raises(TypeError, match="Supply"):
        FinitePlayers(latent_n_players=3)


@pytest.mark.parametrize("latent_count", [0, 5])
def test_empty_ids_allow_an_explicit_latent_space(latent_count):
    players = FinitePlayers(jnp.empty((0,), dtype=int), latent_n_players=latent_count)
    assert players.n_players == 0
    assert players.latent_n_players == latent_count


@pytest.mark.parametrize("metadata", [None, list("abcdefghij")])
def test_batch_operations_preserve_the_latent_space_and_metadata(metadata):
    players = FinitePlayers(jnp.array([[0, 2], [1, 7]]), players=metadata)
    count = players.latent_n_players
    for updated in [players[0], players[:, None], players.reshape((1, 2))]:
        assert updated.latent_n_players == count
        assert updated.players is players.players
    copied = replace(players, ids=jnp.array([0, 1]))
    assert copied.latent_n_players == count
    assert copied.players is players.players
    with pytest.raises(ValueError, match=r"latent_n_players|metadata catalog"):
        replace(players, ids=jnp.array([count]))


@pytest.mark.parametrize("metadata", [None, list("abcdefghij")])
def test_removing_highest_and_all_ids_preserves_latent_count(metadata):
    players = FinitePlayers(jnp.array([0, 3, 7]), players=metadata)
    reduced = players.remove(jnp.array([7]))
    assert reduced.ids.tolist() == [0, 3]
    assert reduced.latent_n_players == players.latent_n_players
    assert reduced.players is players.players
    empty = reduced.remove(jnp.array([0, 3]))
    assert empty.n_players == 0
    assert empty.latent_n_players == players.latent_n_players
    assert empty.players is players.players


@pytest.mark.parametrize("packed", [False, True])
def test_expand_maps_local_positions_to_latent_ids_and_zero_fills(packed):
    players = FinitePlayers(jnp.array([4, 1]), latent_n_players=7)
    coalition = jnp.array([True, False])  # positions follow sorted IDs [1, 4]
    if packed:
        coalition = pack_bool_coalition(coalition)
    expanded = players.expand_coalition(coalition)
    assert expanded.dtype == (jnp.uint8 if packed else jnp.bool_)
    membership = unpack_bool_coalition(expanded, 7) if packed else expanded
    assert membership.tolist() == [False, True, False, False, False, False, False]
    assert players.is_valid_latent(expanded)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    ("player_batch", "coalition_batch", "expected_batch"),
    [
        ((), (3,), (3,)),
        ((1,), (3,), (3,)),
        ((2, 1), (3,), (2, 3)),
        ((3,), (2, 1), (2, 3)),
        ((2, 3), (1, 2, 3), (1, 2, 3)),
        ((1,), (0,), (0,)),
        ((0, 1), (3,), (0, 3)),
    ],
)
def test_expand_coalition_broadcasts_symmetrically(
    packed, player_batch, coalition_batch, expected_batch
):
    ids = jnp.broadcast_to(jnp.array([1, 4]), (*player_batch, 2))
    players = FinitePlayers(ids, latent_n_players=7)
    coalition = jnp.broadcast_to(jnp.array([True, False]), (*coalition_batch, 2))
    if packed:
        coalition = pack_bool_coalition(coalition)
    expanded = players.expand_coalition(coalition)
    if packed:
        expanded = unpack_bool_coalition(expanded, 7)
    expected = jnp.broadcast_to(
        jnp.array([False, True, False, False, False, False, False]), (*expected_batch, 7)
    )
    assert expanded.shape == (*expected_batch, 7)
    assert bool(jnp.array_equal(expanded, expected))


@pytest.mark.parametrize("packed", [False, True])
def test_expansion_broadcasts_different_sets_and_selections_together(packed):
    players = FinitePlayers(jnp.array([[[1, 4]], [[0, 3]]]), latent_n_players=6)
    coalition = jnp.array([[True, False], [False, True], [True, True]])
    if packed:
        coalition = pack_bool_coalition(coalition)
    expanded = players.expand_coalition(coalition)
    if packed:
        expanded = unpack_bool_coalition(expanded, 6)
    assert expanded.shape == (2, 3, 6)
    assert expanded.tolist() == [
        [
            [False, True, False, False, False, False],
            [False, False, False, False, True, False],
            [False, True, False, False, True, False],
        ],
        [
            [True, False, False, False, False, False],
            [False, False, False, True, False, False],
            [True, False, False, True, False, False],
        ],
    ]


@pytest.mark.parametrize("n_players", [0, 1, 7, 8, 9, 16, 17])
def test_packed_expansion_crosses_byte_boundaries(n_players):
    latent_count = 2 * n_players + 3
    ids = 2 * jnp.arange(n_players) + 1
    players = FinitePlayers(ids, latent_n_players=latent_count)
    coalition = jnp.arange(n_players) % 2 == 0
    packed = players.expand_coalition(pack_bool_coalition(coalition))
    expected = jnp.zeros(latent_count, dtype=bool).at[ids].set(coalition)
    assert packed.shape == ((latent_count + 7) // 8,)
    assert bool(jnp.array_equal(unpack_bool_coalition(packed, latent_count), expected))
    assert not bool(jnp.any(jnp.unpackbits(packed, bitorder="little")[latent_count:]))


def test_packed_expansion_ignores_input_padding_and_clears_output_padding():
    players = FinitePlayers(jnp.array([3]), latent_n_players=13)
    expanded = players.expand_coalition(jnp.array([255], dtype=jnp.uint8))
    assert expanded.tolist() == [8, 0]


@pytest.mark.parametrize("packed", [False, True])
def test_duplicate_ids_combine_membership_by_or(packed):
    players = FinitePlayers(jnp.array([2, 2, 5]), latent_n_players=6)
    coalition = jnp.array([[True, False, True], [False, True, False]])
    if packed:
        coalition = pack_bool_coalition(coalition)
    expanded = players.expand_coalition(coalition)
    if packed:
        expanded = unpack_bool_coalition(expanded, 6)
    assert expanded.tolist() == [
        [False, False, True, False, False, True],
        [False, False, True, False, False, False],
    ]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    ("player_batch", "coalition_batch", "n_players", "latent_count"),
    [((), (), 0, 0), ((), (3,), 0, 5), ((0,), (), 2, 0), ((0, 1), (3,), 0, 7)],
)
def test_expansion_handles_empty_sets_and_batches(
    packed, player_batch, coalition_batch, n_players, latent_count
):
    players = FinitePlayers(
        jnp.zeros((*player_batch, n_players), dtype=int), latent_n_players=latent_count
    )
    coalition = jnp.zeros((*coalition_batch, n_players), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    expanded = players.expand_coalition(coalition)
    if packed:
        expanded = unpack_bool_coalition(expanded, latent_count)
    batch = jnp.broadcast_shapes(player_batch, coalition_batch)
    assert expanded.shape == (*batch, latent_count)
    assert not bool(jnp.any(expanded))


@pytest.mark.parametrize("packed", [False, True])
def test_expansion_and_latent_validation_support_jit(packed):
    players = FinitePlayers(jnp.array([[1, 3], [0, 2]]), players=list("abcdefg"))
    coalition = jnp.array([[True, False], [True, True]])
    if packed:
        coalition = pack_bool_coalition(coalition)
    expand = jax.jit(lambda candidate, selections: candidate.expand_coalition(selections))
    expanded = expand(players, coalition)
    assert bool(jnp.array_equal(expanded, players.expand_coalition(coalition)))
    validate = jax.jit(lambda candidate, selections: candidate.is_valid_latent(selections))
    assert bool(validate(players, expanded))
    restored = jax.jit(lambda candidate: candidate)(players)
    assert restored.latent_n_players == 7


@pytest.mark.parametrize(
    "coalition",
    [jnp.ones((2,)), jnp.ones((3,), dtype=bool), jnp.ones((2,), dtype=jnp.uint8), [True, False]],
)
def test_expand_checks_local_coalition_type_and_width(coalition):
    players = FinitePlayers(jnp.array([1, 4]), latent_n_players=7)
    with pytest.raises(TypeCheckError):
        players.expand_coalition(coalition)


def test_expand_rejects_incompatible_batches():
    players = FinitePlayers(jnp.array([[1, 3], [0, 2]]), latent_n_players=7)
    with pytest.raises(ValueError, match="Inadmissible"):
        players.expand_coalition(jnp.ones((3, 2), dtype=bool))


@pytest.mark.parametrize("latent_count", [0, 1, 7, 8, 9, 16, 17])
@pytest.mark.parametrize("packed", [False, True])
def test_latent_validity_uses_latent_width(latent_count, packed):
    players = FinitePlayers(jnp.empty((0,), dtype=int), latent_n_players=latent_count)
    coalition = jnp.ones((2, latent_count), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    assert players.is_valid_latent(coalition)


def test_latent_validity_does_not_validate_membership_or_padding_values():
    players = FinitePlayers(jnp.array([1, 3]), latent_n_players=7)
    assert players.is_valid_latent(jnp.ones((7,), dtype=bool))
    assert players.is_valid_latent(jnp.array([255], dtype=jnp.uint8))
    assert players.is_valid(jnp.ones((2,), dtype=bool))
    assert not players.is_valid_latent(jnp.ones((2,), dtype=bool))
    assert not players.is_valid(jnp.ones((7,), dtype=bool))


@pytest.mark.parametrize("packed", [False, True])
def test_latent_validity_respects_game_batch_context(packed):
    players = FinitePlayers(jnp.array([1, 3]), latent_n_players=7)
    coalition = jnp.ones((3, 7), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    assert players.is_valid_latent(coalition)
    assert not players.is_valid_latent(coalition, batch_shape=(2,))
    assert players.is_valid_latent(coalition, batch_shape=(3,))


@pytest.mark.parametrize(
    "coalition",
    [
        None,
        [True] * 9,
        np.ones((9,), dtype=bool),
        jnp.ones((9,)),
        jnp.ones((8,), dtype=bool),
        jnp.ones((1,), dtype=jnp.uint8),
        jnp.ones((), dtype=bool),
    ],
)
def test_latent_validity_rejects_bad_storage(coalition):
    players = FinitePlayers(jnp.array([1, 3]), latent_n_players=9)
    assert not players.is_valid_latent(coalition)
