"""Finite player sets with optional, shared player metadata."""

from dataclasses import dataclass, field
from math import prod
from typing import ClassVar, Literal, TypeIs, overload, override

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Int, UInt8
from ndlike.array_like import NumpyLike
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.coalition import FiniteCoalition
from shapiq.coalition.coalition import (
    BoolCoalition,
    PackedBoolCoalition,
    pack_bool_coalition,
    unpack_bool_coalition,
)
from shapiq.typing import typechecked

from .latent import LatentPlayers
from .players import Players


@typechecked
def _validate_ids(ids: Int[jax.Array, "*batch n_players"]) -> Int[jax.Array, "*batch n_players"]:
    """Validate positional ID arrays independently of the broader catalog protocols."""
    return ids


@jax.jit
def _remove_sorted_ids(ids: jax.Array, removed_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Locate removals by binary search and compact each sorted player set."""
    n_players = ids.shape[-1]
    n_removed = removed_ids.shape[-1]
    n_remaining = n_players - n_removed

    def remove_row(row: jax.Array, removals: jax.Array) -> tuple[jax.Array, jax.Array]:
        removals = jnp.sort(removals, stable=False)
        positions = jnp.searchsorted(row, removals, side="left")
        found = (positions < n_players) & (jnp.take(row, positions, mode="clip") == removals)
        unique = (positions == n_players - 1) | (
            jnp.take(row, positions + 1, mode="clip") != removals
        )
        valid = jnp.all(found & unique) & jnp.all(removals[1:] != removals[:-1])
        removed = jnp.zeros(n_players, dtype=bool).at[positions].set(True, mode="drop")
        # Fixed-size nonzero uses linear compaction rather than sorting players.
        indices = jnp.nonzero(~removed, size=n_remaining)[0]
        return row[indices], valid

    n_sets = prod(ids.shape[:-1])
    remaining, valid = jax.vmap(remove_row)(
        ids.reshape((n_sets, n_players)), removed_ids.reshape((n_sets, n_removed))
    )
    return remaining.reshape((*ids.shape[:-1], n_remaining)), jnp.all(valid)


@typechecked
def _remove_ids(
    ids: Int[jax.Array, "#*batch n_players"],
    removed_ids: Int[jax.Array, "#*batch n_removed"],
) -> Int[jax.Array, "*batch n_players-n_removed"]:
    """Validate and remove IDs from sorted, fixed-width player sets."""
    n_players = ids.shape[-1]
    n_removed = removed_ids.shape[-1]
    if n_removed > n_players:
        msg = "Cannot remove more IDs than there are players in each set."
        raise ValueError(msg)
    batch_shape = jnp.broadcast_shapes(ids.shape[:-1], removed_ids.shape[:-1])
    ids = jnp.broadcast_to(ids, (*batch_shape, n_players))
    removed_ids = jnp.broadcast_to(removed_ids, (*batch_shape, n_removed))
    if n_removed == 0:
        return ids
    remaining, valid = _remove_sorted_ids(ids, removed_ids)
    if not bool(valid):
        msg = "Removed IDs must be distinct and occur exactly once in every player set."
        raise ValueError(msg)
    return remaining


def _infer_latent_count(ids: jax.Array, latent_n_players: int | None, *, has_metadata: bool) -> int:
    """Infer or validate the latent count using the sorted sets' endpoints."""
    if latent_n_players is not None and latent_n_players < 0:
        msg = "latent_n_players must be nonnegative."
        raise ValueError(msg)
    if not ids.size:
        return 0 if latent_n_players is None else latent_n_players
    if latent_n_players is None:
        minimum, maximum = jax.device_get(
            jnp.stack((jnp.min(ids[..., 0]), jnp.max(ids[..., -1])))
        ).tolist()
        invalid = minimum < 0
        latent_n_players = maximum + 1
    else:
        invalid_ids = ids[..., 0] < 0
        # Counts may exceed the ID dtype's maximum (e.g. int8 IDs and count 128).
        # In that case all representable nonnegative IDs already fit; avoid a
        # narrowing scalar conversion that could overflow during comparison.
        if latent_n_players <= int(jnp.iinfo(ids.dtype).max):
            invalid_ids |= ids[..., -1] >= latent_n_players
        invalid = bool(jnp.any(invalid_ids))
    if invalid:
        msg = (
            "Player IDs must index the metadata catalog: 0 <= ids < len(players)."
            if has_metadata
            else "Player IDs must satisfy 0 <= ids < latent_n_players."
        )
        raise ValueError(msg)
    return latent_n_players


@typechecked
def _expand_bool_coalition(
    ids: Int[jax.Array, "#*batch n_players"],
    coalition: Bool[jax.Array, "#*batch n_players"],
    latent_n_players: int,
) -> Bool[jax.Array, "*batch {latent_n_players}"]:
    """Scatter local membership into latent ID positions with batch broadcasting."""
    batch_shape = jnp.broadcast_shapes(ids.shape[:-1], coalition.shape[:-1])
    ids = jnp.broadcast_to(ids, (*batch_shape, ids.shape[-1]))
    coalition = jnp.broadcast_to(coalition, (*batch_shape, coalition.shape[-1]))

    def expand_row(row_ids: jax.Array, membership: jax.Array) -> jax.Array:
        # Repeated IDs denote the same latent player: membership combines by OR.
        latent = jnp.zeros(latent_n_players, dtype=bool)
        # Nonnegative IDs need no wrapping, which can overflow narrow ID dtypes
        # when the latent dimension itself is not representable in that dtype.
        return latent.at[row_ids].max(membership, mode="drop", wrap_negative_indices=False)

    n_sets = prod(batch_shape)
    expanded = jax.vmap(expand_row)(
        ids.reshape((n_sets, ids.shape[-1])),
        coalition.reshape((n_sets, coalition.shape[-1])),
    )
    return expanded.reshape((*batch_shape, latent_n_players))


def _use_sparse_packed_expansion(
    n_players: int,
    latent_n_players: int,
    batch_shape: tuple[int, ...],
    id_itemsize: int,
) -> bool:
    """Checks whether a packed expansion is likely to be more memory-efficient than a boolean expansion.

    The boolean route (_expand_bool_coalition) needs local and latent membership arrays.
    Budget the packed route's (_expand_packed_bits) explicit intermediates:
    four ID-sized buffers (selected IDs, prefix, shifted prefix, destination byte indices);
    three source-bit byte buffers; three selection flags; four destination-bit byte buffers;
    three shared position/offset index vectors plus byte offsets; and scan seeds.
    """
    n_sets = prod(batch_shape)
    if n_players == 0 or n_sets == 0:
        return True
    bool_size = jnp.dtype(jnp.bool_).itemsize
    bool_bytes = n_sets * (n_players + latent_n_players) * bool_size

    byte_size = jnp.dtype(jnp.uint8).itemsize
    index_size = jax.dtypes.canonicalize_dtype(int).itemsize

    n_local = n_sets * n_players
    scan_and_scatter_bytes = n_local * 4 * id_itemsize
    source_bit_bytes = n_local * 3 * byte_size
    selection_bytes = n_local * 3 * bool_size
    destination_bit_bytes = n_local * 4 * byte_size
    shared_index_bytes = n_players * (3 * index_size + byte_size)
    seed_bytes = n_sets * id_itemsize
    packed_bytes = (
        scan_and_scatter_bytes
        + source_bit_bytes
        + selection_bytes
        + destination_bit_bytes
        + shared_index_bytes
        + seed_bytes
    )
    return packed_bytes < bool_bytes


@jax.jit(static_argnames=("latent_n_players",))
def _expand_packed_bits(ids: jax.Array, coalition: jax.Array, latent_n_players: int) -> jax.Array:
    """Scatter packed bits without allocating a boolean latent coalition."""
    batch_shape = jnp.broadcast_shapes(ids.shape[:-1], coalition.shape[:-1])
    n_players = ids.shape[-1]
    n_sets = prod(batch_shape)
    n_bytes = (latent_n_players + 7) // 8
    if n_players == 0 or n_sets == 0:
        return jnp.zeros((*batch_shape, n_bytes), dtype=jnp.uint8)

    ids = jnp.broadcast_to(ids, (*batch_shape, n_players))
    positions = jnp.arange(n_players)
    bits = (
        jnp.take(coalition, positions >> 3, axis=-1) >> (positions & 7).astype(jnp.uint8)
    ) & jnp.uint8(1)
    selected = jnp.broadcast_to(bits != 0, (*batch_shape, n_players))

    # IDs are sorted and nonnegative. Keep only the first *selected* occurrence
    # of each ID, including runs whose first occurrence is unselected. Without
    # this scan, adding duplicate masks could carry into another player's bit.
    last_selected = jnp.maximum.accumulate(jnp.where(selected, ids, -1), axis=-1)
    previous = jnp.concatenate(
        (jnp.full((*batch_shape, 1), -1, dtype=ids.dtype), last_selected[..., :-1]), axis=-1
    )
    unique_selected = selected & (ids != previous)
    updates = jnp.where(
        unique_selected,
        jnp.uint8(1) << (ids.astype(jnp.uint8) & jnp.uint8(7)),
        jnp.uint8(0),
    )

    def scatter_row(row_ids: jax.Array, row_updates: jax.Array) -> jax.Array:
        # Selected masks are disjoint, so addition is equivalent to OR and each
        # byte's sum is at most 255. Unreferenced and padding bits remain zero.
        latent = jnp.zeros(n_bytes, dtype=jnp.uint8)
        return latent.at[row_ids >> 3].add(row_updates, mode="drop", wrap_negative_indices=False)

    expanded = jax.vmap(scatter_row)(
        ids.reshape((n_sets, n_players)), updates.reshape((n_sets, n_players))
    )
    return expanded.reshape((*batch_shape, n_bytes))


@typechecked
def _expand_packed_coalition(
    ids: Int[jax.Array, "#*batch n_players"],
    coalition: UInt8[jax.Array, "#*batch (n_players+7)//8"],
    latent_n_players: int,
) -> UInt8[jax.Array, "*batch ({latent_n_players}+7)//8"]:
    """Expand packed membership with checked widths and symmetric broadcasting."""
    return _expand_packed_bits(ids, coalition, latent_n_players)


@jax.jit
@typechecked
def _contract_coalition(
    ids: Int[jax.Array, "#*batch n_players"],
    coalition: Bool[jax.Array, "#*batch _"] | UInt8[jax.Array, "#*batch _"],
) -> Bool[jax.Array, "*batch n_players"] | UInt8[jax.Array, "*batch (n_players+7)//8"]:
    """Gather local membership, extracting only referenced bits for packed input."""
    batch_shape = jnp.broadcast_shapes(ids.shape[:-1], coalition.shape[:-1])
    n_players = ids.shape[-1]
    packed = coalition.dtype == jnp.uint8
    width = (n_players + 7) // 8 if packed else n_players
    if n_players == 0 or prod(batch_shape) == 0:
        return jnp.zeros((*batch_shape, width), dtype=coalition.dtype)

    ids = jnp.broadcast_to(ids, (*batch_shape, n_players))
    coalition = jnp.broadcast_to(coalition, (*batch_shape, coalition.shape[-1]))
    indices = ids >> 3 if packed else ids
    # Construction already ensures IDs are nonnegative latent indices.
    selected = jnp.take_along_axis(coalition, indices, axis=-1, mode="promise_in_bounds")
    if packed:
        bits = (selected >> (ids.astype(jnp.uint8) & jnp.uint8(7))) & jnp.uint8(1)
        return jnp.packbits(bits, axis=-1, bitorder="little")
    return selected


def _is_valid_coalition(
    coalition: object,
    n_players: int,
    players_batch: tuple[int, ...],
    batch_shape: tuple[int, ...] | None,
) -> TypeIs[FiniteCoalition]:
    """Check coalition dtype, width, and logical broadcasting without reading values."""
    if not isinstance(coalition, jax.Array) or coalition.ndim == 0:
        return False
    if coalition.dtype == jnp.bool_:
        width = n_players
    elif coalition.dtype == jnp.uint8:
        width = (n_players + 7) // 8
    else:
        return False
    if coalition.shape[-1] != width:
        return False
    shape = players_batch if batch_shape is None else batch_shape
    try:
        if jnp.broadcast_shapes(players_batch, shape) != shape:
            return False
        jnp.broadcast_shapes(shape, coalition.shape[:-1])
    except ValueError:
        return False
    return True


@typechecked
@dataclass(slots=True, init=False, frozen=True)
class FinitePlayers[P](Players[FiniteCoalition], JaxAxisProtected[jax.Array]):
    """A batch of equally sized finite player sets.

    ``ids`` has shape ``(*batch, n_players)`` and is sorted in ascending order
    along its final axis at construction unless ``sorted=True`` promises it is
    already sorted. This promise is not checked. Coalition membership refers to
    positions in this sorted order, not to the IDs themselves or the original
    input order. Different sets may reference different IDs while retaining
    the same number of players. IDs are not renumbered or deduplicated.

    ``latent_n_players`` is the size of the shared, zero-based ID space, which
    can include players absent from every current set. It is the metadata
    catalog length when supplied, otherwise ``max(ids) + 1`` for explicit IDs
    (zero for an empty array), or the generated player count. An explicit
    ``latent_n_players`` with IDs or a local count can reserve additional
    unreferenced players. A redundant count with metadata must match its length.
    IDs must satisfy ``0 <= ids < latent_n_players``.

    ``players`` is an optional, shared catalog: ``players[id]`` contains the
    metadata for that ID. It may contain unreferenced entries, and its length
    need not equal ``n_players``. Indexing and reshaping the player-set batch
    leave this catalog unchanged. Raw metadata is wrapped in ``LatentPlayers``;
    an existing catalog is reused. Its numeric storage can be packed, while
    arbitrary objects retain their Python identity. The catalog is opaque,
    identity-keyed static metadata under JAX, not dynamic numerical input.

    Pass an ID array, a metadata catalog/list/tuple, or an integer count as a single positional
    argument, or use ``ids=``, ``players=``, or ``n_players=``. IDs and metadata
    may also be supplied together, in either positional order or by keyword.
    Without explicit IDs, sequential IDs are generated and the batch shape is
    ``()``. Lists, tuples, catalogs, and NumPy arrays always denote metadata;
    positional JAX arrays denote IDs. Use ``players=`` for JAX array metadata.
    A positional integer always denotes ``n_players``; ``latent_n_players`` is
    keyword-only. A local player count cannot be combined with IDs or metadata.

    Args:
        *args: Positional IDs, metadata, or a player count, distinguished by type.
        ids: Integer player IDs, with a final player axis even for empty sets.
        players: Optional metadata indexed by nonnegative player ID. Its length
            must be at least ``max(ids) + 1`` when any IDs are present.
        n_players: Nonnegative player count, used only without IDs or metadata.
        latent_n_players: Optional, keyword-only shared ID-space size. Must
            exceed every ID, be at least ``n_players`` for generated IDs, and
            equal ``len(players)`` when metadata is given. Explicit counts skip
            max-based inference but retain ID bounds validation.
        sorted: Whether explicit IDs are already sorted along their final axis.
            Defaults to ``False``, which sorts them. ``True`` skips sorting but
            retains metadata bounds validation; the caller guarantees ordering.
    """

    ids: Int[jax.Array, "*batch n_players"]
    players: LatentPlayers[P] | None = None
    latent_n_players: int = field(kw_only=True)
    n_players: int = field(init=False)
    protected_axes: ClassVar[dict[str, int]] = {"ids": 1}

    @overload
    def __init__(
        self,
        ids: Int[jax.Array, "*batch n_players"],
        players: LatentPlayers[P]
        | list[P]
        | tuple[P, ...]
        | np.ndarray
        | jax.Array
        | JaxLike
        | NumpyLike
        | None = None,
        *,
        latent_n_players: int | None = None,
        sorted: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        players: LatentPlayers[P] | list[P] | tuple[P, ...] | np.ndarray | JaxLike | NumpyLike,
        ids: Int[jax.Array, "*batch n_players"] | None = None,
        *,
        latent_n_players: int | None = None,
        sorted: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        n_players: int,
        *,
        latent_n_players: int | None = None,
        sorted: bool = False,
    ) -> None: ...

    def __init__(
        self,
        *args: Int[jax.Array, "*batch n_players"]
        | LatentPlayers[P]
        | list[P]
        | tuple[P, ...]
        | np.ndarray
        | JaxLike
        | NumpyLike
        | int
        | None,
        ids: Int[jax.Array, "*batch n_players"] | None = None,
        players: LatentPlayers[P]
        | list[P]
        | tuple[P, ...]
        | np.ndarray
        | jax.Array
        | JaxLike
        | NumpyLike
        | None = None,
        n_players: int | None = None,
        latent_n_players: int | None = None,
        sorted: bool = False,  # noqa: A002 - public constructor flag
    ) -> None:
        """Normalize positional inputs and validate metadata coverage."""
        if len(args) > 2:
            msg = "At most two positional arguments may be supplied."
            raise TypeError(msg)
        for argument in args:
            if argument is None:
                continue
            if isinstance(argument, (LatentPlayers, list, tuple, np.ndarray)) or (
                not isinstance(argument, jax.Array) and isinstance(argument, (JaxLike, NumpyLike))
            ):
                if players is not None:
                    msg = "players was supplied more than once."
                    raise TypeError(msg)
                players = argument
            elif isinstance(argument, int):
                if n_players is not None:
                    msg = "n_players was supplied more than once."
                    raise TypeError(msg)
                n_players = argument
            else:
                if ids is not None:
                    msg = "ids was supplied more than once."
                    raise TypeError(msg)
                ids = argument

        explicit_ids = ids is not None
        if explicit_ids:
            ids = _validate_ids(ids)
        if n_players is not None:
            if ids is not None or players is not None:
                msg = "n_players cannot be combined with ids or players."
                raise TypeError(msg)
            if n_players < 0:
                msg = "n_players must be nonnegative."
                raise ValueError(msg)
            if latent_n_players is None:
                latent_n_players = n_players
            elif latent_n_players < n_players:
                msg = "latent_n_players must be at least n_players."
                raise ValueError(msg)
            ids = jnp.arange(n_players)
        elif ids is None:
            if players is None:
                msg = "Supply ids, players, or n_players."
                raise TypeError(msg)
            ids = jnp.arange(len(players))
        elif not sorted:
            ids = jnp.sort(ids, axis=-1, stable=False)

        if players is not None:
            if latent_n_players is not None and latent_n_players != len(players):
                msg = "latent_n_players must equal the metadata catalog length len(players)."
                raise ValueError(msg)
            latent_n_players = len(players)
        if explicit_ids:
            latent_n_players = _infer_latent_count(
                ids, latent_n_players, has_metadata=players is not None
            )

        if players is not None and not isinstance(players, LatentPlayers):
            players = LatentPlayers(players)

        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "players", players)
        object.__setattr__(self, "n_players", ids.shape[-1])
        object.__setattr__(
            self,
            "latent_n_players",
            ids.shape[-1] if latent_n_players is None else latent_n_players,
        )

    @typechecked
    def remove(
        self,
        removed_ids: Int[jax.Array, "*removed_batch n_removed"],
    ) -> "FinitePlayers[P]":
        """Return player sets with the requested IDs removed from each set.

        ``removed_ids`` always has a final removal axis, including for a single
        ID. Its leading axes broadcast symmetrically with this object's batch.
        A vector of shape ``(n_removed,)`` applies the same removals to every
        set; batched removals can also produce multiple reduced sets from one
        shared player set. Only the leading batch axes participate in broadcasting.

        Each requested ID must be distinct and occur exactly once in its set.
        Surviving IDs retain their sorted order, and the shared metadata catalog
        and ``latent_n_players`` are unchanged. An empty removal axis leaves the
        sets unchanged but still applies batch broadcasting.
        With ``n`` players and ``k`` removals, each set takes
        ``O(n + k log(n) + k log(k))`` work and ``O(n + k)`` storage; no Cartesian
        match array is formed. The result is already sorted and is not re-sorted.

        Validation inspects array values, so removal requires concrete arrays
        and should be performed outside JIT-compiled functions.

        Args:
            removed_ids: Integer IDs with shape ``(#*batch, n_removed)``.

        Returns:
            A new batch with shape
            ``broadcast_shapes(self.shape, removed_ids.shape[:-1])`` and
            ``self.n_players - n_removed`` players per set.

        Raises:
            ValueError: Too many IDs are requested, IDs are repeated or absent,
                or a requested ID occurs more than once in a set.
        """
        ids = _remove_ids(self.ids, removed_ids)
        return FinitePlayers(
            ids, players=self.players, latent_n_players=self.latent_n_players, sorted=True
        )

    @overload
    def expand_coalition(
        self,
        coalition: BoolCoalition,
        *,
        packed_expansion: bool | Literal["auto"] = "auto",
    ) -> BoolCoalition: ...

    @overload
    def expand_coalition(
        self,
        coalition: PackedBoolCoalition,
        *,
        packed_expansion: bool | Literal["auto"] = "auto",
    ) -> PackedBoolCoalition: ...

    @typechecked
    def expand_coalition(
        self,
        coalition: Bool[jax.Array, "... {self.n_players}"]
        | UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"],
        *,
        packed_expansion: bool | Literal["auto"] = "auto",
    ) -> (
        Bool[jax.Array, "... {self.latent_n_players}"]
        | UInt8[jax.Array, "... bytes=({self.latent_n_players}+7)//8"]
    ):
        """Expand local membership into latent ID positions, zero elsewhere.

        Coalition and player-set batches broadcast symmetrically. Boolean input
        returns boolean output of width ``latent_n_players``; packed ``uint8``
        input returns packed output of width ``ceil(latent_n_players / 8)``.
        Duplicate IDs combine membership by logical OR.

        With ``packed_expansion="auto"``, packed input uses direct bit scattering
        when its logical working-buffer volume is smaller than local/latent
        boolean membership, and a boolean intermediate otherwise.

        Args:
            coalition: Admissible local boolean or packed ``uint8`` membership.
            packed_expansion: ``True`` forces direct packed expansion; ``False``
                forces boolean expansion, unpacking/repacking packed input;
                ``"auto"`` uses the logical-storage rule for packed input.

        Raises:
            ValueError: The coalition batch is inadmissible, or
                ``packed_expansion=True`` is requested for boolean input.
        """
        if not self.is_valid(coalition):
            msg = "Inadmissible local coalition batch."
            raise ValueError(msg)
        packed = coalition.dtype == jnp.uint8
        if packed_expansion is True and not packed:
            msg = "packed_expansion=True requires a packed uint8 coalition."
            raise ValueError(msg)
        if packed and packed_expansion is not False:
            batch_shape = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
            if packed_expansion is True or _use_sparse_packed_expansion(
                self.n_players, self.latent_n_players, batch_shape, self.ids.dtype.itemsize
            ):
                return _expand_packed_coalition(self.ids, coalition, self.latent_n_players)
        membership = unpack_bool_coalition(coalition, self.n_players) if packed else coalition
        expanded = _expand_bool_coalition(self.ids, membership, self.latent_n_players)
        return pack_bool_coalition(expanded) if packed else expanded

    @overload
    def contract_coalition(self, coalition: BoolCoalition) -> BoolCoalition: ...

    @overload
    def contract_coalition(self, coalition: PackedBoolCoalition) -> PackedBoolCoalition: ...

    @typechecked
    def contract_coalition(
        self,
        coalition: Bool[jax.Array, "... {self.latent_n_players}"]
        | UInt8[jax.Array, "... bytes=({self.latent_n_players}+7)//8"],
    ) -> (
        Bool[jax.Array, "... {self.n_players}"]
        | UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ):
        """Restrict latent membership to each player set's local positions.

        Coalition and player-set batches broadcast symmetrically. Local position
        ``i`` receives membership at latent ID ``self.ids[..., i]``. Selected
        latent players absent from a set are dropped; duplicate IDs repeat the
        same membership at every matching local position. Consequently, this
        reverses expansion for unique IDs, but cannot recover different local
        selections that expansion combined for duplicate IDs.

        Boolean input returns boolean output of width ``n_players``. Packed
        ``uint8`` input returns little-endian packed output of width
        ``ceil(n_players / 8)``, gathering only referenced latent bytes rather
        than unpacking the latent coalition. Input padding is ignored and output
        padding is zero. Supports JIT on constructed player batches.

        Args:
            coalition: Admissible latent boolean or packed ``uint8`` membership.

        Raises:
            ValueError: The latent coalition batch is inadmissible.
        """
        if not self.is_valid_latent(coalition):
            msg = "Inadmissible latent coalition batch."
            raise ValueError(msg)
        return _contract_coalition(self.ids, coalition)

    @typechecked
    @override
    def is_valid(
        self,
        coalition: object,
        *,
        batch_shape: tuple[int, ...] | None = None,
    ) -> TypeIs[FiniteCoalition]:
        """Check unpacked boolean or packed uint8 coalitions and batch broadcasting.

        Packed coalitions use ``ceil(n_players / 8)`` bytes along their final
        axis. Only static dtype and shape metadata is inspected; padding bits
        in the final byte do not affect admissibility.
        """
        return _is_valid_coalition(coalition, self.n_players, self.shape, batch_shape)

    @typechecked
    def is_valid_latent(
        self,
        coalition: object,
        *,
        batch_shape: tuple[int, ...] | None = None,
    ) -> TypeIs[FiniteCoalition]:
        """Check latent coalition dtype, width, and batch broadcasting only.

        Values are not inspected: coalitions may select IDs absent from the
        current sets, and packed padding bits need not be zero. ``batch_shape``
        has the same meaning as in :meth:`is_valid`.
        """
        return _is_valid_coalition(coalition, self.latent_n_players, self.shape, batch_shape)
