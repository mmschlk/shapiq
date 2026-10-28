"""Finite player batches, shared metadata, and coalition admissibility."""

from dataclasses import FrozenInstanceError, dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.players import FinitePlayers, LatentPlayers, Players


@pytest.mark.parametrize("n_players", [0, 1, 9])
@pytest.mark.parametrize("keyword", [False, True])
def test_count_creates_an_unbatched_set_without_metadata(n_players, keyword):
    players = FinitePlayers(n_players=n_players) if keyword else FinitePlayers(n_players)
    assert players.shape == ()
    assert players.n_players == n_players
    assert players.latent_n_players == n_players
    assert players.players is None
    assert players.ids.tolist() == list(range(n_players))
    assert jnp.issubdtype(players.ids.dtype, jnp.integer)


@pytest.mark.parametrize("n_players", [-1, -3])
@pytest.mark.parametrize("keyword", [False, True])
def test_rejects_negative_counts(n_players, keyword):
    with pytest.raises(ValueError, match="nonnegative"):
        if keyword:
            FinitePlayers(n_players=n_players)
        else:
            FinitePlayers(n_players)


@pytest.mark.parametrize("n_players", [1.5, "3", ["a"]])
def test_rejects_noninteger_keyword_counts(n_players):
    with pytest.raises(TypeCheckError):
        FinitePlayers(n_players=n_players)


@pytest.mark.parametrize("metadata", [[], ["a"], ["a", "b", "c"], [object(), object()], [4, 9]])
@pytest.mark.parametrize("keyword", [False, True])
def test_metadata_creates_an_unbatched_set_with_sequential_ids(metadata, keyword):
    players = FinitePlayers(players=metadata) if keyword else FinitePlayers(metadata)
    assert players.shape == ()
    assert players.n_players == len(metadata)
    assert players.latent_n_players == len(metadata)
    assert isinstance(players.players, LatentPlayers)
    assert list(players.players) == metadata
    assert players.ids.tolist() == list(range(len(metadata)))
    assert jnp.issubdtype(players.ids.dtype, jnp.integer)


@pytest.mark.parametrize("metadata", ["ab", 3, {"a": 1}])
def test_rejects_unsupported_keyword_metadata(metadata):
    with pytest.raises(TypeCheckError):
        FinitePlayers(players=metadata)


@pytest.mark.parametrize("keyword", [False, True])
def test_ids_only_preserve_ids_and_omit_metadata(keyword):
    ids = jnp.array([[2, 7], [4, 9]])
    players = FinitePlayers(ids=ids) if keyword else FinitePlayers(ids)
    assert bool(jnp.array_equal(players.ids, ids))
    assert players.players is None
    assert players.shape == (2,)
    assert players.n_players == 2
    assert players.latent_n_players == 10


def test_optional_positional_arguments_can_be_none():
    ids = jnp.array([2, 0])
    players = FinitePlayers(ids, None)
    assert players.ids.tolist() == [0, 2]
    assert ids.tolist() == [2, 0]
    assert players.players is None
    metadata = ["a", "b"]
    players = FinitePlayers(metadata, None)
    assert players.ids.tolist() == [0, 1]
    assert list(players.players) == metadata


@pytest.mark.parametrize("form", ["positional", "reversed", "keywords", "mixed", "reversed-mixed"])
def test_ids_and_metadata_support_positional_and_keyword_forms(form):
    ids = jnp.array([[0, 2], [3, 1]])
    metadata = ["a", "b", "c", "d", "unused"]
    if form == "positional":
        players = FinitePlayers(ids, metadata)
    elif form == "reversed":
        players = FinitePlayers(metadata, ids)
    elif form == "keywords":
        players = FinitePlayers(ids=ids, players=metadata)
    elif form == "mixed":
        players = FinitePlayers(ids, players=metadata)
    else:
        players = FinitePlayers(metadata, ids=ids)
    assert players.ids.tolist() == [[0, 2], [1, 3]]
    assert list(players.players) == metadata
    assert players.shape == (2,)
    assert players.n_players == 2


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("options", [{}, {"sorted": False}])
def test_constructor_sorts_each_set_without_renumbering_or_deduplicating(keyword, options):
    ids = jnp.array([[[9, 1, 4, 1], [7, 3, 2, 8]], [[6, 0, 5, 2], [8, 4, 1, 3]]])
    original = ids.tolist()
    players = FinitePlayers(ids=ids, **options) if keyword else FinitePlayers(ids, **options)
    assert players.shape == (2, 2)
    assert players.n_players == 4
    assert bool(jnp.array_equal(players.ids, jnp.sort(ids, axis=-1)))
    assert players.ids[0, 0].tolist() == [1, 1, 4, 9]
    assert ids.tolist() == original


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("metadata", [None, ["a", "b", "c", "d"]])
def test_sorted_true_skips_sorting_and_retains_the_id_array(monkeypatch, keyword, metadata):
    ids = jnp.array([[0, 2], [1, 3]])

    def unexpected_sort(*args, **kwargs):
        msg = "sorted=True must not sort the player IDs"
        raise AssertionError(msg)

    monkeypatch.setattr(jnp, "sort", unexpected_sort)
    players = (
        FinitePlayers(ids=ids, players=metadata, sorted=True)
        if keyword
        else FinitePlayers(ids, players=metadata, sorted=True)
    )
    assert players.ids is ids
    assert (None if players.players is None else list(players.players)) == metadata
    assert players.shape == (2,)
    assert players.n_players == 2


@pytest.mark.parametrize("argument", [0, 3, [], ["a", "b"]])
def test_sorted_true_also_accepts_count_and_metadata_construction(argument):
    players = FinitePlayers(argument, sorted=True)
    count = len(argument) if isinstance(argument, list) else argument
    assert players.ids.tolist() == list(range(count))
    assert players.n_players == count
    assert players.shape == ()


@pytest.mark.parametrize("ids", [[-1, 0], [0, 3], [[0, 1], [2, 3]]])
def test_sorted_true_still_validates_metadata_bounds(ids):
    with pytest.raises(ValueError, match="metadata catalog"):
        FinitePlayers(jnp.array(ids), players=["a", "b", "c"], sorted=True)


@pytest.mark.parametrize("flag", [0, 1, "yes", None, jnp.ones((), dtype=bool)])
def test_sorted_flag_must_be_boolean(flag):
    with pytest.raises(TypeCheckError):
        FinitePlayers(jnp.arange(3), sorted=flag)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((), {}, "Supply"),
        ((None,), {}, "Supply"),
        ((), {"ids": None, "players": None, "n_players": None}, "Supply"),
        ((1, 1), {}, "n_players was supplied more than once"),
        ((1,), {"n_players": 1}, "n_players was supplied more than once"),
        (([], []), {}, "players was supplied more than once"),
        (([],), {"players": []}, "players was supplied more than once"),
        ((jnp.arange(2), jnp.arange(2)), {}, "ids was supplied more than once"),
        ((jnp.arange(2),), {"ids": jnp.arange(2)}, "ids was supplied more than once"),
        ((0, []), {}, "cannot be combined"),
        ((), {"n_players": 0, "players": []}, "cannot be combined"),
        ((), {"n_players": 0, "ids": jnp.arange(0)}, "cannot be combined"),
        (([],), {"n_players": 0}, "cannot be combined"),
        ((jnp.arange(0),), {"n_players": 0}, "cannot be combined"),
        ((jnp.arange(0), [], 0), {}, "At most two"),
    ],
)
def test_rejects_missing_duplicate_and_conflicting_inputs(args, kwargs, message):
    with pytest.raises(TypeError, match=message):
        FinitePlayers(*args, **kwargs)


@pytest.mark.parametrize("argument", [1.5, "ab", {"a": 1}])
def test_rejects_unsupported_positional_inputs(argument):
    with pytest.raises(TypeCheckError):
        FinitePlayers(argument)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("n_players", [0, 1, 9])
def test_logical_shape_excludes_player_axis(batch_shape, n_players):
    players = FinitePlayers(jnp.zeros((*batch_shape, n_players), dtype=int))
    assert isinstance(players, Players)
    assert players.shape == batch_shape
    assert players.ndim == len(batch_shape)
    assert players.size == np.prod(batch_shape)
    assert players.n_players == n_players
    assert players.players is None


@pytest.mark.parametrize(
    "ids",
    [jnp.arange(3, dtype=jnp.float32), jnp.array(3)],
    ids=["wrong-dtype", "missing-player-axis"],
)
@pytest.mark.parametrize("keyword", [False, True])
def test_rejects_invalid_id_storage(ids, keyword):
    with pytest.raises(TypeCheckError):
        if keyword:
            FinitePlayers(ids=ids)
        else:
            FinitePlayers(ids)


def test_lists_are_metadata_not_keyword_id_arrays():
    with pytest.raises(TypeCheckError):
        FinitePlayers(ids=[0, 1, 2])


def test_metadata_catalog_is_shared_and_may_include_unused_players():
    metadata = LatentPlayers(["zero", "one", "two", "three", "four", "unused"])
    players = FinitePlayers[str](jnp.array([[0, 4], [3, 1]]), players=metadata)
    assert players.shape == (2,)
    assert players.n_players == 2
    assert players.players is metadata
    assert players[0].players is metadata
    assert players[1].players is metadata
    assert players[0].ids.tolist() == [0, 4]
    assert players[1].ids.tolist() == [1, 3]
    assert players.reshape((1, 2)).players is metadata
    assert players[:, None].players is metadata
    assert players[:, None].shape == (2, 1)


@pytest.mark.parametrize("field", ["ids", "players", "n_players", "latent_n_players"])
def test_dataclass_fields_are_frozen(field):
    players = FinitePlayers(["a", "b", "c"])
    with pytest.raises(FrozenInstanceError):
        setattr(players, field, getattr(players, field))
    with pytest.raises(FrozenInstanceError):
        delattr(players, field)


def test_dataclass_replace_preserves_the_catalog_and_latent_count():
    players = FinitePlayers(["a", "b", "c"])
    updated = replace(players, ids=jnp.array([2, 0]))
    assert updated.ids.tolist() == [0, 2]
    assert updated.n_players == 2
    assert updated.latent_n_players == players.latent_n_players
    assert updated.players is players.players
    with pytest.raises(ValueError, match="metadata catalog"):
        replace(players, ids=jnp.array([3]))


def test_metadata_retains_arbitrary_python_objects():
    @dataclass
    class Metadata:
        name: str

    metadata = [Metadata("first"), Metadata("second")]
    players = FinitePlayers[Metadata](jnp.array([1]), players=metadata)
    assert players.players[1] is metadata[1]


def test_pytree_metadata_is_opaque_and_preserves_catalog_identity():
    metadata = LatentPlayers([{"array": np.arange(2)}, jnp.arange(3), ["nested"]])
    players = FinitePlayers(jnp.array([[0, 2], [1, 2]]), players=metadata)
    leaves, structure = jax.tree.flatten(players)
    assert len(leaves) == 1
    assert leaves[0] is players.ids
    restored = jax.tree.unflatten(structure, leaves)
    assert restored.players is metadata
    assert restored.latent_n_players == 3
    identity = jax.jit(lambda candidate: candidate)
    assert identity(players).players is metadata
    mapped = jax.vmap(lambda candidate: candidate)(players)
    assert mapped.players is metadata
    assert bool(jnp.array_equal(mapped.ids, players.ids))


def test_jit_cache_keys_catalog_identity_but_keeps_ids_dynamic():
    traces = []

    @jax.jit
    def get_ids(candidate):
        traces.append(None)
        return candidate.ids

    metadata = LatentPlayers(["a", "b", "c"])
    first = FinitePlayers(jnp.array([0, 1]), players=metadata)
    second = FinitePlayers(jnp.array([1, 2]), players=metadata)
    assert get_ids(first).tolist() == [0, 1]
    assert get_ids(second).tolist() == [1, 2]
    assert len(traces) == 1
    get_ids(FinitePlayers(jnp.array([0, 1]), players=list(metadata))).block_until_ready()
    assert len(traces) == 2


@pytest.mark.parametrize("ids", [[0, 3], [[0, 1], [3, 2]], [-1, 0]])
def test_metadata_must_cover_every_referenced_id(ids):
    with pytest.raises(ValueError, match="metadata catalog"):
        FinitePlayers(jnp.array(ids), players=["a", "b", "c"])


def test_metadata_may_be_omitted_for_noncontiguous_ids():
    players = FinitePlayers(jnp.array([[2, 7], [4, 9]]))
    assert players.n_players == 2
    assert players.is_valid(jnp.ones((2,), dtype=bool))


@pytest.mark.parametrize("ids_shape", [(0,), (2, 0), (0, 3)])
def test_empty_id_arrays_allow_an_empty_catalog(ids_shape):
    players = FinitePlayers(jnp.zeros(ids_shape, dtype=int), players=[])
    assert players.n_players == ids_shape[-1]
    assert isinstance(players.players, LatentPlayers)
    assert not players.players


def test_nonempty_id_arrays_reject_an_empty_catalog():
    with pytest.raises(ValueError, match="metadata catalog"):
        FinitePlayers(jnp.array([0]), players=[])


@pytest.mark.parametrize("metadata", [None, list("abcdefghij")])
def test_remove_uses_ids_preserves_order_and_retains_metadata(metadata):
    players = FinitePlayers(jnp.array([7, 2, 9, 4]), players=metadata)
    result = players.remove(removed_ids=jnp.array([4, 2]))
    assert isinstance(result, FinitePlayers)
    assert result is not players
    assert result.shape == ()
    assert result.n_players == 2
    assert result.ids.tolist() == [7, 9]
    assert result.players is players.players
    assert players.ids.tolist() == [2, 4, 7, 9]
    assert players.n_players == 4


def test_remove_one_id_per_player_set_requires_a_removal_axis():
    players = FinitePlayers(jnp.array([[2, 4, 8], [1, 3, 7]]))
    result = players.remove(jnp.array([[4], [7]]))
    assert result.shape == (2,)
    assert result.n_players == 2
    assert result.ids.tolist() == [[2, 8], [1, 3]]


def test_remove_multiple_ids_per_player_set():
    players = FinitePlayers(jnp.array([[8, 2, 4, 6], [7, 3, 1, 5]]))
    result = players.remove(jnp.array([[6, 2], [1, 7]]))
    assert result.shape == (2,)
    assert result.n_players == 2
    assert result.ids.tolist() == [[4, 8], [3, 5]]


@pytest.mark.parametrize("removed_batch", [(), (1,), (1, 1), (3,), (2, 1), (1, 3), (2, 3)])
def test_shared_removals_broadcast_into_player_batches(removed_batch):
    ids = jnp.broadcast_to(jnp.array([3, 0, 2, 1]), (2, 3, 4))
    players = FinitePlayers(ids)
    removed_ids = jnp.broadcast_to(jnp.array([0, 1]), (*removed_batch, 2))
    result = players.remove(removed_ids)
    assert result.shape == (2, 3)
    assert result.n_players == 2
    assert result.ids.tolist() == [[[2, 3]] * 3] * 2


@pytest.mark.parametrize("metadata", [None, ["a", "b", "c", "d"]])
def test_batched_removals_broadcast_one_player_set_into_multiple_results(metadata):
    players = FinitePlayers(jnp.arange(4), players=metadata)
    result = players.remove(jnp.array([[0], [1], [3]]))
    assert result.shape == (3,)
    assert result.n_players == 3
    assert result.ids.tolist() == [[1, 2, 3], [0, 2, 3], [0, 1, 2]]
    assert result.players is players.players
    assert players.shape == ()
    assert players.ids.tolist() == [0, 1, 2, 3]


def test_multiple_removals_can_vary_for_one_shared_player_set():
    players = FinitePlayers(5)
    result = players.remove(jnp.array([[0, 4], [1, 2]]))
    assert result.shape == (2,)
    assert result.n_players == 3
    assert result.ids.tolist() == [[1, 2, 3], [0, 3, 4]]


def test_both_player_and_removal_batches_can_expand():
    players = FinitePlayers(jnp.array([[[0, 1, 2, 4]], [[0, 1, 2, 5]]]))
    result = players.remove(jnp.array([[0], [1], [2]]))
    assert result.shape == (2, 3)
    assert result.n_players == 3
    assert result.ids.tolist() == [
        [[1, 2, 4], [0, 2, 4], [0, 1, 4]],
        [[1, 2, 5], [0, 2, 5], [0, 1, 5]],
    ]
    assert players.shape == (2, 1)


@pytest.mark.parametrize(
    ("player_batch", "removed_batch", "expected_batch"),
    [
        ((), (3,), (3,)),
        ((1,), (3,), (3,)),
        ((2, 1), (3,), (2, 3)),
        ((3,), (2, 1), (2, 3)),
        ((2, 3), (1, 2, 3), (1, 2, 3)),
        ((1,), (0,), (0,)),
        ((), (0, 3), (0, 3)),
        ((0, 1), (3,), (0, 3)),
        ((1, 3), (0, 1), (0, 3)),
    ],
)
def test_remove_uses_the_symmetric_broadcast_batch(player_batch, removed_batch, expected_batch):
    ids = jnp.broadcast_to(jnp.arange(4), (*player_batch, 4))
    players = FinitePlayers(ids)
    removed_ids = jnp.broadcast_to(jnp.array([1, 3]), (*removed_batch, 2))
    result = players.remove(removed_ids)
    assert result.shape == expected_batch
    assert result.n_players == 2
    assert bool(
        jnp.array_equal(result.ids, jnp.broadcast_to(jnp.array([0, 2]), (*expected_batch, 2)))
    )


@pytest.mark.parametrize("n_players", [0, 3])
def test_empty_removals_still_expand_the_player_batch(n_players):
    metadata = list(range(n_players))
    players = FinitePlayers(metadata)
    result = players.remove(jnp.empty((2, 3, 0), dtype=int))
    assert result is not players
    assert result.shape == (2, 3)
    assert result.n_players == n_players
    assert result.players is players.players
    assert bool(jnp.array_equal(result.ids, jnp.broadcast_to(players.ids, (2, 3, n_players))))


def test_batched_removal_of_all_players_expands_to_empty_sets():
    players = FinitePlayers(["a", "b", "c"])
    result = players.remove(jnp.array([[0, 1, 2], [2, 0, 1]]))
    assert result.shape == (2,)
    assert result.n_players == 0
    assert result.ids.shape == (2, 0)
    assert result.players is players.players


@pytest.mark.parametrize("removed_ids", [[[0], [4]], [[0, 1], [2, 2]]])
def test_broadcast_removal_rejects_an_invalid_request_in_any_result(removed_ids):
    players = FinitePlayers(4)
    with pytest.raises(ValueError, match="distinct and occur exactly once"):
        players.remove(jnp.array(removed_ids))


def test_shared_removal_vector_is_never_a_squeezed_single_id_batch():
    players = FinitePlayers(jnp.array([[0, 1, 2], [2, 0, 1]]))
    result = players.remove(jnp.array([0, 2]))
    assert result.n_players == 1
    assert result.ids.tolist() == [[1], [1]]


def test_broadcast_removals_can_vary_over_one_batch_axis():
    ids = jnp.broadcast_to(jnp.array([0, 1, 2, 3]), (2, 3, 4))
    players = FinitePlayers(ids)
    result = players.remove(jnp.array([[[1]], [[3]]]))
    assert result.shape == (2, 3)
    assert result.n_players == 3
    assert result.ids.tolist() == [[[0, 2, 3]] * 3, [[0, 1, 2]] * 3]


@pytest.mark.parametrize("batch_shape", [(), (2,), (0,), (2, 0)])
@pytest.mark.parametrize("n_players", [0, 1, 3])
def test_removing_no_ids_returns_a_new_unchanged_batch(batch_shape, n_players):
    ids = jnp.broadcast_to(jnp.arange(n_players), (*batch_shape, n_players))
    metadata = list(range(n_players))
    players = FinitePlayers(ids, players=metadata)
    result = players.remove(jnp.array([], dtype=int))
    assert result is not players
    assert result.shape == batch_shape
    assert result.n_players == n_players
    assert result.players is players.players
    assert bool(jnp.array_equal(result.ids, players.ids))


@pytest.mark.parametrize("batch_shape", [(), (2,), (0,), (2, 0)])
def test_removing_all_ids_returns_empty_player_sets(batch_shape):
    ids = jnp.broadcast_to(jnp.array([2, 0, 1]), (*batch_shape, 3))
    metadata = ["a", "b", "c", "unused"]
    players = FinitePlayers(ids, players=metadata)
    result = players.remove(jnp.array([0, 1, 2]))
    assert result.shape == batch_shape
    assert result.n_players == 0
    assert result.ids.shape == (*batch_shape, 0)
    assert result.players is players.players


@pytest.mark.parametrize(
    ("ids", "removed_ids"),
    [
        ([0, 1, 2], [3]),
        ([[0, 1, 2], [0, 3, 4]], [1]),
        ([0, 1, 2], [1, 1]),
        ([[0, 1, 2], [0, 1, 2]], [[0, 1], [1, 1]]),
        ([0, 0, 1], [0]),
        ([0, 1, 2], [-1]),
    ],
    ids=["absent", "absent-in-one-set", "duplicate", "batched-duplicate", "ambiguous", "negative"],
)
def test_remove_rejects_ids_that_cannot_be_removed_exactly_once(ids, removed_ids):
    players = FinitePlayers(jnp.array(ids))
    with pytest.raises(ValueError, match="distinct and occur exactly once"):
        players.remove(jnp.array(removed_ids))


def test_unselected_duplicate_ids_do_not_prevent_removal():
    players = FinitePlayers(jnp.array([1, 1, 2, 3]))
    result = players.remove(jnp.array([2]))
    assert result.ids.tolist() == [1, 1, 3]


@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize(("n_players", "n_removed"), [(1, 1), (2, 1), (17, 8), (32, 31), (32, 32)])
def test_sorted_removal_matches_a_reference_for_sparse_ids(batch_shape, n_players, n_removed):
    rng = np.random.default_rng(42)
    n_sets = int(np.prod(batch_shape))
    ids = np.stack([rng.choice(np.arange(256), n_players, replace=False) for _ in range(n_sets)])
    removals = np.stack([rng.choice(row, n_removed, replace=False) for row in ids])
    expected = np.stack(
        [np.sort(row[~np.isin(row, removed)]) for row, removed in zip(ids, removals, strict=True)]
    )
    players = FinitePlayers(jnp.array(ids.reshape((*batch_shape, n_players))))
    result = players.remove(jnp.array(removals.reshape((*batch_shape, n_removed))))
    assert result.shape == batch_shape
    assert result.n_players == n_players - n_removed
    assert np.array_equal(result.ids, expected.reshape((*batch_shape, n_players - n_removed)))


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.int16, jnp.int32])
@pytest.mark.parametrize("remove_maximum", [False, True])
def test_removal_handles_integer_limits_without_id_sentinels(dtype, remove_maximum):
    limits = jnp.iinfo(dtype)
    players = FinitePlayers(jnp.array([limits.max, 1, 0], dtype=dtype))
    removed = limits.max if remove_maximum else 0
    result = players.remove(jnp.array([removed], dtype=dtype))
    assert result.ids.tolist() == ([0, 1] if remove_maximum else [1, limits.max])
    assert result.ids.dtype == dtype
    assert result.latent_n_players == int(limits.max) + 1


def test_remove_uses_the_normal_constructor_with_sorted_true(monkeypatch):
    metadata = ["a", "b", "c", "d"]
    players = FinitePlayers(jnp.array([3, 1, 0, 2]), players=metadata)
    constructor = FinitePlayers.__init__
    calls = []

    def record_constructor(self, *args, **kwargs):
        calls.append(kwargs)
        constructor(self, *args, **kwargs)

    monkeypatch.setattr(FinitePlayers, "__init__", record_constructor)
    result = players.remove(jnp.array([2, 0]))
    assert calls == [{"players": players.players, "latent_n_players": 4, "sorted": True}]
    assert result.ids.tolist() == [1, 3]
    assert result.n_players == 2
    assert result.players is players.players


@pytest.mark.parametrize("n_players", [0, 2])
def test_remove_rejects_more_ids_than_players(n_players):
    players = FinitePlayers(n_players)
    with pytest.raises(ValueError, match="more IDs"):
        players.remove(jnp.arange(n_players + 1))


@pytest.mark.parametrize(
    "removed_ids",
    [
        jnp.array(1),
        jnp.array([1.0]),
        jnp.array([True]),
        jnp.array([1], dtype=jnp.uint8),
        np.array([1]),
        [1],
        1,
        None,
        jnp.zeros((4, 1), dtype=int),
    ],
    ids=[
        "missing-removal-axis",
        "float",
        "bool",
        "uint8",
        "numpy",
        "list",
        "python-int",
        "none",
        "incompatible-batch",
    ],
)
def test_remove_checks_types_and_batch_shapes(removed_ids):
    players = FinitePlayers(jnp.broadcast_to(jnp.arange(3), (2, 3, 3)))
    with pytest.raises(TypeCheckError):
        players.remove(removed_ids)


@pytest.mark.parametrize("n_players", [0, 1, 7, 8, 9, 16, 17])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1)])
@pytest.mark.parametrize("packed", [False, True])
def test_accepts_packed_and_unpacked_coalitions(n_players, batch_shape, packed):
    players = FinitePlayers(jnp.arange(n_players))
    coalition = jnp.ones((*batch_shape, n_players), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    assert players.is_valid(coalition)
    assert coalition in players


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    ("players_batch", "coalition_batch", "batch_shape", "valid"),
    [
        ((2, 1), (3,), None, True),
        ((2,), (3,), None, False),
        ((3,), (2, 1), (2, 3), True),
        ((3,), (5, 1, 1), (2, 3), True),
        ((3,), (4,), (2, 3), False),
        ((), (3,), (2,), False),
        ((2,), (), (1,), False),
        ((1, 1), (), (1,), False),
    ],
)
def test_batch_broadcasting(players_batch, coalition_batch, batch_shape, valid, packed):
    players = FinitePlayers(jnp.zeros((*players_batch, 9), dtype=int))
    coalition = jnp.ones((*coalition_batch, 9), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    assert players.is_valid(coalition, batch_shape=batch_shape) is valid


@pytest.mark.parametrize(
    "coalition",
    [
        jnp.ones((9,)),
        jnp.ones((9,), dtype=jnp.int32),
        jnp.ones((2,), dtype=jnp.uint16),
        jnp.ones((8,), dtype=bool),
        jnp.ones((1,), dtype=jnp.uint8),
        jnp.ones((3,), dtype=jnp.uint8),
        jnp.ones((), dtype=bool),
        jnp.ones((), dtype=jnp.uint8),
        np.ones((9,), dtype=bool),
        [True] * 9,
        None,
    ],
    ids=[
        "float",
        "int",
        "uint16",
        "wrong-bool-width",
        "too-few-bytes",
        "too-many-bytes",
        "bool-scalar",
        "packed-scalar",
        "numpy",
        "list",
        "none",
    ],
)
def test_invalid_coalitions_return_false(coalition):
    players = FinitePlayers(jnp.arange(9))
    assert players.is_valid(coalition) is False


def test_packed_validation_does_not_inspect_padding_bits():
    players = FinitePlayers(jnp.arange(1))
    assert players.is_valid(jnp.array([255], dtype=jnp.uint8))


@pytest.mark.parametrize("metadata", [None, ["a", "b", "c"]])
@pytest.mark.parametrize("packed", [False, True])
def test_validation_supports_jit_and_player_pytrees(metadata, packed):
    players = FinitePlayers(jnp.array([[0, 2], [1, 0]]), players=metadata)
    coalition = jnp.ones((2, 2), dtype=bool)
    if packed:
        coalition = pack_bool_coalition(coalition)
    validate = jax.jit(lambda candidate, selections: candidate.is_valid(selections))
    assert bool(validate(players, coalition))
    assert not bool(validate(players, jnp.ones((3, 9), dtype=bool)))
    restored = jax.jit(lambda candidate: candidate)(players)
    assert restored.players is players.players
    assert restored.n_players == 2
    assert restored.latent_n_players == 3
    assert bool(jnp.array_equal(restored.ids, players.ids))
