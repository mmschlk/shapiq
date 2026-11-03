"""Read-only sequence behavior and packed/object latent player storage."""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from ndlike.jax_like import JaxLikeImplementation
from ndlike.protected_axis.jax import JaxAxisProtected
from ndlike.protected_axis.numpy import NumpyAxisProtected

from shapiq.players import FinitePlayers, LatentPlayers


@dataclass
class _JaxRows(JaxAxisProtected[jax.Array]):
    __hash__ = None
    left: jax.Array
    right: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and np.array_equal(self.left, other.left)
            and np.array_equal(self.right, other.right)
        )


@dataclass
class _NumpyRows(NumpyAxisProtected[np.ndarray]):
    __hash__ = None
    left: np.ndarray
    right: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and np.array_equal(self.left, other.left)
            and np.array_equal(self.right, other.right)
        )


@pytest.mark.parametrize("source", [[], ["a", "b", "a"], ("a", "b", "a")])
def test_object_catalog_is_a_read_only_sequence(source):
    catalog = LatentPlayers(source)
    assert isinstance(catalog, Sequence)
    assert len(catalog) == len(source)
    assert bool(catalog) == bool(source)
    assert list(catalog) == list(source)
    assert list(reversed(catalog)) == list(reversed(source))
    assert catalog._data.dtype == object
    assert catalog._data.ndim == 1
    assert "missing" not in catalog
    assert catalog.count("missing") == 0
    with pytest.raises(ValueError, match="not in"):
        catalog.index("missing")


def test_object_storage_is_shallow_and_does_not_expand_nested_entries():
    entries = [[1, 2], {"name": "a"}, object()]
    catalog = LatentPlayers(entries)
    assert catalog._data.shape == (3,)
    for index, entry in enumerate(entries):
        assert catalog[index] is entry
    entries[0] = "replacement"
    assert catalog[0] == [1, 2]
    catalog[1]["name"] = "updated"  # Entries, unlike the catalog, are not deeply frozen.
    assert entries[1]["name"] == "updated"


@pytest.mark.parametrize(
    "source",
    [
        np.arange(12).reshape(3, 4),
        np.arange(24).reshape(3, 8)[:, ::2],
        jnp.arange(12).reshape(3, 4),
    ],
)
def test_numeric_array_inputs_keep_axis_zero_indexing(source):
    catalog = LatentPlayers(source)
    assert len(catalog) == 3
    assert catalog._data.shape == (3, 4)
    assert catalog._data.dtype.kind in "iu"
    assert np.array_equal(catalog[1], np.asarray(source)[1])
    if isinstance(catalog._data, jax.Array):
        assert isinstance(catalog[0], jax.Array)
    assert np.array_equal(list(catalog), source)


@pytest.mark.parametrize("source", [[1, 2, 3], [True, False], [1.0, 2.5], [1j, 2j]])
def test_numeric_scalar_lists_always_use_object_storage(source):
    catalog = LatentPlayers(source)
    assert isinstance(catalog._data, np.ndarray)
    assert catalog._data.ndim == 1
    assert catalog._data.dtype == object
    assert list(catalog) == source
    for index, player in enumerate(source):
        assert catalog[index] is player


def test_numeric_array_lists_preserve_objects_without_promoting_dtypes():
    source = [np.array([1], dtype=np.int16), np.array([2.5], dtype=np.float64)]
    catalog = LatentPlayers(source)
    assert catalog._data.shape == (2,)
    assert catalog._data.dtype == object
    assert catalog[0] is source[0]
    assert catalog[1] is source[1]


@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0), (2, 0, 4)])
def test_empty_numeric_dimensions_are_preserved(shape):
    catalog = LatentPlayers(np.empty(shape))
    assert len(catalog) == shape[0]
    assert catalog._data.shape == shape
    assert catalog[:]._data.shape == shape


@pytest.mark.parametrize(
    "entries",
    [
        [np.arange(2), np.arange(2)],
        [jnp.arange(2), jnp.arange(2)],
        [np.arange(2), np.arange(3)],
        [np.arange(2), "label"],
        [np.array(["a"]), np.array(["b"])],
        [np.array([1], dtype=object), np.array([2], dtype=object)],
    ],
)
def test_array_lists_always_use_one_dimensional_object_storage(entries):
    catalog = LatentPlayers(entries)
    assert catalog._data.shape == (len(entries),)
    assert catalog._data.dtype == object
    for index, entry in enumerate(entries):
        assert catalog[index] is entry


@pytest.mark.parametrize(
    "source",
    [np.array(["a", "b"]), np.array([{"a": 1}], dtype=object), np.array([1, 2], dtype=object)],
)
def test_nonnumeric_array_inputs_use_object_storage(source):
    catalog = LatentPlayers(source)
    assert catalog._data.dtype == object
    assert list(catalog) == list(source)


@pytest.mark.parametrize("source", [np.array(1), jnp.array(1)])
def test_scalar_arrays_are_not_catalogs(source):
    with pytest.raises(ValueError, match="leading player dimension"):
        LatentPlayers(source)


@pytest.mark.parametrize("packed", [False, True])
def test_integer_and_slice_indexing_follow_sequence_semantics(packed):
    source = np.arange(12).reshape(4, 3) if packed else list("abcd")
    catalog = LatentPlayers(source)
    assert np.array_equal(catalog[np.int64(-1)], source[-1])
    for selection in [slice(None), slice(1, 3), slice(None, None, -1), slice(9, 10)]:
        selected = catalog[selection]
        assert isinstance(selected, LatentPlayers)
        assert selected is not catalog
        assert np.array_equal(list(selected), list(source[selection]))
        assert type(selected._data) is type(catalog._data)
    assert np.shares_memory(catalog._data, catalog[1:]._data)
    for index in [len(catalog), -len(catalog) - 1]:
        with pytest.raises(IndexError):
            catalog[index]
    with pytest.raises(ValueError, match="step"):
        catalog[::0]


@pytest.mark.parametrize("backend", ["object", "numpy", "jax"])
@pytest.mark.parametrize("mask_type", [list, np.asarray, jnp.asarray])
def test_boolean_filters_return_catalogs(backend, mask_type):
    source = np.arange(12).reshape(4, 3)
    if backend == "object":
        source = list("abcd")
    elif backend == "jax":
        source = jnp.asarray(source)
    catalog = LatentPlayers(source)
    for mask in [[True, False, True, False], [False] * 4, [True] * 4]:
        selected = catalog[mask_type(mask)]
        assert isinstance(selected, LatentPlayers)
        assert len(selected) == sum(mask)
        assert np.array_equal(selected._data, catalog._data[np.asarray(mask)])
        assert type(selected._data) is type(catalog._data)


@pytest.mark.parametrize(
    "mask",
    [np.ones((), dtype=bool), np.ones((2, 2), dtype=bool), np.array([0, 1]), [0, 1], [True]],
)
def test_filters_must_be_one_dimensional_boolean_masks_of_the_correct_length(mask):
    catalog = LatentPlayers(list("ab"))
    with pytest.raises((IndexError, TypeCheckError)):
        catalog[mask]


def test_empty_boolean_filter_of_empty_catalog():
    catalog = LatentPlayers([])
    assert len(catalog[np.array([], dtype=bool)]) == 0


def test_read_only_list_queries_support_start_and_stop_bounds():
    catalog = LatentPlayers(list("abaca"))
    assert "a" in catalog
    assert catalog.count("a") == 3
    assert catalog.index("a") == 0
    assert catalog.index("a", 0, None) == 0
    assert catalog.index("a", 0, 3) == 0
    assert catalog.index("a", 1) == 2
    assert catalog.index("a", 1, None) == 2
    assert catalog.index("a", -2) == 4
    assert catalog.index("a", np.int64(1), np.int64(3)) == 2
    with pytest.raises(ValueError, match="not in"):
        catalog.index("a", 1, 2)


@pytest.mark.parametrize("backend", ["object", "numpy", "jax"])
def test_array_player_queries_use_whole_array_equality(backend):
    entries = [np.array([1, 2]), np.array([3, 4]), np.array([1, 2])]
    source = entries
    if backend == "numpy":
        source = np.stack(entries)
    elif backend == "jax":
        source = jnp.asarray(entries)
    catalog = LatentPlayers(source)
    player = np.array([1, 2])
    assert player in catalog
    assert catalog.count(player) == 2
    assert catalog.index(player) == 0
    assert catalog.index(player, 1) == 2
    assert np.array([1]) not in catalog
    assert np.array([8, 9]) not in catalog


@pytest.mark.parametrize("writeable", [False, True])
def test_numpy_backing_and_indexed_views_are_reused_without_flag_changes(writeable):
    source = np.arange(12).reshape(3, 4)
    source.flags.writeable = writeable
    catalog = LatentPlayers(source)
    assert catalog._data is source
    assert source.flags.writeable == writeable
    assert np.shares_memory(catalog[0], source)
    assert np.shares_memory(catalog[:]._data, source)
    assert catalog[0].flags.writeable == writeable
    assert catalog[:]._data.flags.writeable == writeable
    # Boolean indexing allocates through NumPy itself, without a defensive copy.
    assert not np.shares_memory(catalog[[True, False, True]]._data, source)
    with pytest.raises(TypeError):
        catalog[0] = np.zeros(4)
    with pytest.raises(TypeError):
        del catalog[0]
    with pytest.raises(FrozenInstanceError):
        catalog._data = source
    with pytest.raises(FrozenInstanceError):
        del catalog._data


def test_no_mutation_or_array_operations_are_exposed():
    catalog = LatentPlayers([1, 2])
    for name in [
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "sort",
        "reverse",
        "reshape",
    ]:
        assert not hasattr(catalog, name)
    with pytest.raises(TypeError):
        _ = catalog + catalog
    with pytest.raises(TypeError):
        _ = catalog * 2
    with pytest.raises(TypeError):
        _ = catalog < LatentPlayers([1, 2])


def test_catalog_hash_and_equality_are_identity_based():
    first = LatentPlayers([{"unhashable": np.arange(2)}])
    second = LatentPlayers(list(first))
    assert hash(first) == id(first)
    alias = first
    assert first == alias
    assert first != second
    assert first != list(first)
    assert {first: "first", second: "second"}[first] == "first"


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [LatentPlayers(list("abc")), list("abc"), tuple("abc"), np.arange(6).reshape(3, 2)],
)
def test_finite_players_normalizes_raw_metadata_and_reuses_catalogs(metadata, keyword):
    players = FinitePlayers(players=metadata) if keyword else FinitePlayers(metadata)
    assert isinstance(players.players, LatentPlayers)
    assert len(players.players) == 3
    assert players.ids.tolist() == [0, 1, 2]
    if isinstance(metadata, LatentPlayers):
        assert players.players is metadata
    assert players[None].players is players.players
    assert players.remove(jnp.array([1])).players is players.players
    assert FinitePlayers.tree_flatten is JaxLikeImplementation.tree_flatten
    restored = jax.jit(lambda candidate: candidate)(players)
    assert restored.players is players.players


def test_jax_array_metadata_is_explicit_to_avoid_ambiguity_with_ids():
    data = jnp.arange(6).reshape(3, 2)
    players = FinitePlayers(players=data)
    assert len(players.players) == 3
    assert np.array_equal(players.players[1], [2, 3])
    assert FinitePlayers(data).players is None


def test_existing_jax_storage_is_reused_and_stays_on_jax():
    source = jnp.arange(12).reshape(3, 4)
    catalog = LatentPlayers(source)
    assert catalog._data is source
    assert isinstance(catalog[1], jax.Array)
    assert isinstance(catalog[1:]._data, jax.Array)
    assert isinstance(catalog[[True, False, True]]._data, jax.Array)
    assert len(list(catalog)) == 3
    assert len(list(reversed(catalog))) == 3
    for index in [3, -4]:
        with pytest.raises(IndexError):
            catalog[index]
    with pytest.raises(TypeError):
        catalog[0][0] = 99
    changed = catalog[0].at[0].set(99)
    assert int(changed[0]) == 99
    assert int(catalog[0][0]) == 0


def test_extended_jax_float_dtypes_remain_native():
    source = jnp.ones((3, 2), dtype=jnp.bfloat16)
    catalog = LatentPlayers(source)
    assert catalog._data is source
    assert catalog[0].dtype == jnp.bfloat16


@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0)])
def test_empty_jax_dimensions_preserve_native_backing_and_sequence_bounds(shape):
    source = jnp.empty(shape)
    catalog = LatentPlayers(source)
    assert catalog._data is source
    assert len(list(catalog)) == shape[0]
    assert isinstance(catalog[:]._data, jax.Array)
    if shape[0] == 0:
        with pytest.raises(IndexError):
            catalog[0]


def test_numpy_numeric_storage_retains_its_backend_and_dtype():
    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    catalog = LatentPlayers(source)
    assert isinstance(catalog._data, np.ndarray)
    assert catalog._data.dtype == np.float64
    assert catalog._data is source


def test_existing_one_dimensional_object_storage_is_reused():
    source = np.array([{"name": "a"}, {"name": "b"}], dtype=object)
    catalog = LatentPlayers(source)
    assert catalog._data is source
    assert catalog[0] is source[0]
    assert np.shares_memory(catalog[:]._data, source)
    assert source.flags.writeable


@pytest.mark.parametrize("wrapper", [_JaxRows, _NumpyRows])
def test_arraylike_wrappers_preserve_logical_axes_and_structured_players(wrapper, monkeypatch):
    array = jnp.asarray if wrapper is _JaxRows else np.asarray
    source = wrapper(array(np.arange(8).reshape(4, 2)), array(np.arange(12).reshape(4, 3)))

    def unexpected_copy(*args, **kwargs):
        msg = "Catalog construction and indexing must not defensively copy wrappers."
        raise AssertionError(msg)

    monkeypatch.setattr(wrapper, "copy", unexpected_copy)
    catalog = LatentPlayers(source)
    assert isinstance(catalog._data, wrapper)
    assert catalog._data is source
    assert len(catalog) == 4
    assert catalog._data.shape == (4,)
    first = catalog[0]
    assert isinstance(first, wrapper)
    assert first.shape == ()
    assert np.array_equal(first.left, [0, 1])
    assert np.array_equal(first.right, [0, 1, 2])
    assert len(list(catalog)) == 4
    assert len(list(reversed(catalog))) == 4
    assert first in catalog
    assert catalog.count(first) == 1
    assert catalog.index(first) == 0
    for selected in [catalog[1:], catalog[[True, False, True, False]]]:
        assert isinstance(selected._data, wrapper)
        assert selected[0] == source[1 if len(selected) == 3 else 0]
    if wrapper is _NumpyRows:
        assert np.shares_memory(first.left, source.left)
        assert np.shares_memory(first.right, source.right)
    players = FinitePlayers(players=catalog._data)
    assert players.players._data is source
    restored = jax.jit(lambda candidate: candidate)(players)
    assert restored.players is players.players


@pytest.mark.parametrize("wrapper", [_JaxRows, _NumpyRows])
def test_wrapped_player_lists_preserve_objects_without_stacking(wrapper):
    array = jnp.asarray if wrapper is _JaxRows else np.asarray
    first = wrapper(array([0, 1]), array([2, 3, 4]))
    second = wrapper(array([5, 6]), array([7, 8, 9]))
    catalog = LatentPlayers([first, second])
    assert isinstance(catalog._data, np.ndarray)
    assert catalog._data.dtype == object
    assert catalog._data.shape == (2,)
    assert len(catalog) == 2
    assert catalog[0] is first
    assert catalog[1] is second
