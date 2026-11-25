"""Keyed random permutations of [0, N) and colex unranking, on integers of any width.

Numbers are stored as little-endian uint32 words along the last axis: ``x[..., 0]`` holds the
lowest 32 bits. Swap-or-not and colex unranking only add, subtract, and compare numbers, so the
words carry them past JAX's native integer width without enabling x64.
"""

from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Int, Key, UInt32

_BITS = 32


def n_words(n_items: int) -> int:
    """Return the number of words that hold every integer up to ``n_items`` inclusive."""
    return max(1, -(-n_items.bit_length() // _BITS))


def from_int(value: int, width: int) -> UInt32[np.ndarray, " width"]:
    """Return the ``width`` words of a nonnegative Python integer."""
    return np.array([(value >> (_BITS * i)) & 0xFFFFFFFF for i in range(width)], dtype=np.uint32)


def from_positions(
    positions: Int[jax.Array, "*shape"], width: int
) -> UInt32[jax.Array, "*shape width"]:
    """Return the words of nonnegative int32 positions."""
    low = positions.astype(jnp.uint32)[..., None]
    return jnp.concatenate([low, jnp.zeros((*positions.shape, width - 1), jnp.uint32)], axis=-1)


def _add(
    a: UInt32[jax.Array, "*shape width"], b: UInt32[jax.Array, "*shape width"]
) -> tuple[UInt32[jax.Array, "*shape width"], Bool[jax.Array, "*shape"]]:
    """Return ``a + b`` modulo ``2**(32 * width)`` and whether it overflowed."""
    a, b = jnp.broadcast_arrays(a, b)
    words, carry = [], jnp.zeros(a.shape[:-1], dtype=bool)
    for i in range(a.shape[-1]):
        partial = a[..., i] + b[..., i]
        total = partial + carry.astype(jnp.uint32)
        carry = (partial < a[..., i]) | (total < partial)
        words.append(total)
    return jnp.stack(words, axis=-1), carry


def _sub(
    a: UInt32[jax.Array, "*shape width"], b: UInt32[jax.Array, "*shape width"]
) -> tuple[UInt32[jax.Array, "*shape width"], Bool[jax.Array, "*shape"]]:
    """Return ``a - b`` modulo ``2**(32 * width)`` and whether it borrowed, i.e. ``a < b``."""
    a, b = jnp.broadcast_arrays(a, b)
    words, borrow = [], jnp.zeros(a.shape[:-1], dtype=bool)
    for i in range(a.shape[-1]):
        partial = a[..., i] - b[..., i]
        total = partial - borrow.astype(jnp.uint32)
        borrow = (a[..., i] < b[..., i]) | (partial < borrow.astype(jnp.uint32))
        words.append(total)
    return jnp.stack(words, axis=-1), borrow


def less(
    a: UInt32[jax.Array, "*shape width"], b: UInt32[jax.Array, "*shape width"]
) -> Bool[jax.Array, "*shape"]:
    """Return ``a < b``."""
    return _sub(a, b)[1]


def add_mod(
    a: UInt32[jax.Array, "*shape width"],
    b: UInt32[jax.Array, "*shape width"],
    n_items: UInt32[jax.Array, "*shape width"],
) -> UInt32[jax.Array, "*shape width"]:
    """Return ``(a + b) mod n_items`` for ``a, b < n_items``."""
    total, carry = _add(a, b)
    reduced, borrow = _sub(total, n_items)
    return jnp.where((carry | ~borrow)[..., None], reduced, total)


def _sub_mod(
    a: UInt32[jax.Array, "*shape width"],
    b: UInt32[jax.Array, "*shape width"],
    n_items: UInt32[jax.Array, "*shape width"],
) -> UInt32[jax.Array, "*shape width"]:
    """Return ``(a - b) mod n_items`` for ``a, b < n_items``."""
    difference, borrow = _sub(a, b)
    return jnp.where(borrow[..., None], _add(difference, n_items)[0], difference)


def _mix(x: UInt32[jax.Array, "*shape"]) -> UInt32[jax.Array, "*shape"]:
    """Scramble 32 bits (Wellons' lowbias32)."""
    x = (x ^ (x >> 16)) * jnp.uint32(0x7FEB352D)
    x = (x ^ (x >> 15)) * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def n_rounds(n_items: int) -> int:
    """Return the swap-or-not rounds for a domain of ``n_items``, about six per bit."""
    return 6 * max(n_items.bit_length(), 4)


def _mask(n_items: int, width: int) -> UInt32[np.ndarray, " width"]:
    """Return the bits below the bit length of ``n_items - 1``: every number below ``n_items``."""
    return from_int((1 << (n_items - 1).bit_length()) - 1, width)


def _reject_below(
    key: Key[jax.Array, ""],
    bound: UInt32[jax.Array, "*shape width"],
    mask: UInt32[jax.Array, "#*shape width"],
) -> UInt32[jax.Array, "*shape width"]:
    """Draw uniform integers below ``bound`` by rejecting candidates of ``mask``'s bits."""

    def redraw(state: tuple) -> tuple:
        key, value, accepted = state
        key, subkey = jax.random.split(key)
        # Draw a candidate below the next power of two, and reject it if it exceeds the bound.
        candidate = jax.random.bits(subkey, bound.shape, jnp.uint32) & mask
        # Accept the candidate for all new draws and keep the old values
        value = jnp.where(accepted[..., None], value, candidate)
        # Prune the candidate if it exceeds the bound, so the loop continues until all draws are accepted.
        return key, value, less(value, bound)

    start = (key, jnp.zeros(bound.shape, jnp.uint32), jnp.zeros(bound.shape[:-1], dtype=bool))
    return jax.lax.while_loop(lambda state: ~jnp.all(state[2]), redraw, start)[1]


def random_below(
    key: Key[jax.Array, ""], n_items: int, shape: tuple[int, ...]
) -> UInt32[jax.Array, "*shape width"]:
    """Draw uniform integers in ``[0, n_items)`` by rejecting draws of its bit length."""
    width = n_words(n_items)
    bound = jnp.broadcast_to(jnp.asarray(from_int(n_items, width)), (*shape, width))
    return _reject_below(key, bound, jnp.asarray(_mask(n_items, width)))


def random_below_each(
    key: Key[jax.Array, ""], n_items: tuple[int, ...], index: Int[jax.Array, "*shape"], width: int
) -> UInt32[jax.Array, "*shape width"]:
    """Draw a uniform integer in ``[0, n_items[i])`` for every entry ``i`` of ``index``.

    Empty domains count as one item, so a stray index can not stall the rejection loop.
    """
    domains = [max(n, 1) for n in n_items]
    bounds = jnp.asarray(np.stack([from_int(n, width) for n in domains]))[index]
    masks = jnp.asarray(np.stack([_mask(n, width) for n in domains]))[index]
    return _reject_below(key, bounds, masks)


def swap_or_not(
    x: UInt32[jax.Array, "*batch n_draws width"],
    n_items: UInt32[jax.Array, "#*batch #n_draws width"],
    strata: Int[jax.Array, "*batch n_draws"],
    round_keys: UInt32[jax.Array, "*batch n_strata n_rounds width"],
    round_seeds: UInt32[jax.Array, "*batch n_strata n_rounds"],
) -> UInt32[jax.Array, "*batch n_draws width"]:
    """Return ``pi(x)`` for the keyed permutation of ``[0, n_items)`` (Hoang, Morris, Rogaway 2012).

    Every stratum owns a permutation of its own domain: draw ``i`` is permuted within
    ``[0, n_items[i])`` with the round keys and seeds of ``strata[i]``. Each round pairs ``x``
    with ``k - x mod n_items`` for a round key ``k`` drawn uniformly below ``n_items``, and
    swaps them if a hash of the pair is odd. Every round is an involution, so their
    composition is a bijection of every stratum's domain for any keys.
    """

    def one_round(r: jax.Array, x: jax.Array) -> jax.Array:
        # Look up each draw's round key and seed in its stratum; r is the traced round counter.
        key = jnp.take_along_axis(round_keys[..., r, :], strata[..., None], axis=-2)
        seed = jnp.take_along_axis(round_seeds[..., r], strata, axis=-1)  # (*batch, n_draws)
        partner = _sub_mod(key, x, n_items)
        larger = jnp.where(less(x, partner)[..., None], partner, x)  # same for x and its partner
        hashed = jnp.broadcast_to(seed, larger.shape[:-1])
        for i in range(larger.shape[-1]):
            hashed = _mix(hashed ^ larger[..., i])
        return jnp.where((hashed >> 31 == 1)[..., None], partner, x)

    # one_round is compiled once for all rounds, done internally in fori_loop.
    return jax.lax.fori_loop(0, round_keys.shape[-2], one_round, x)


def bits(x: UInt32[jax.Array, "*shape width"], n_bits: int) -> Bool[jax.Array, "*shape {n_bits}"]:
    """Return the lowest ``n_bits`` bits of ``x``, lowest first: the coalition a bitmask encodes."""
    shifts = jnp.arange(_BITS, dtype=jnp.uint32)
    unpacked = (x[..., None] >> shifts) & 1  # (*shape, width, 32)
    return unpacked.reshape(*x.shape[:-1], -1)[..., :n_bits].astype(bool)


def colex_unrank(
    index: UInt32[jax.Array, "*shape width"], n_players: int, size: Int[jax.Array, "#*shape"]
) -> Bool[jax.Array, "*shape {n_players}"]:
    """Return the coalition of ``size`` players with colex rank ``index`` < ``C(n_players, size)``.

    The rank of ``{c_1 < ... < c_k}`` is ``sum_i C(c_i, i)``. Walking players from the last, a
    player joins when the rank is at least ``C(j, k)`` for the ``k`` players still missing.
    """
    width = index.shape[-1]
    # table[j,k] = C(j,k) where j is contained in a k-size coalition, when rank >= C(j,k), otherwise j is not contained.
    table = jnp.asarray(
        np.stack(
            [[from_int(comb(j, k), width) for k in range(n_players + 1)] for j in range(n_players)]
        )
    )  # (n_players, n_players + 1, width); C(j, k) = 0 for k > j

    def visit(step: int | Int[jax.Array, ""], state: tuple) -> tuple:
        rank, missing, coalition = state
        # Decide which player to join, from highest to lowest.
        player = n_players - 1 - step
        # Get the threshold, indicating the boundary between needing the player or skipping it.
        threshold = jnp.take(table[player], missing, axis=0)  # (*shape, width)
        remainder, below = _sub(rank, threshold)  # below: rank < threshold
        # Indicates that the player must join
        joins = (missing > 0) & ~below
        # Move the rank down by the threshold if the player joins
        rank = jnp.where(joins[..., None], remainder, rank)
        # Set the player
        coalition = coalition.at[..., player].set(joins)
        return rank, missing - joins, coalition

    missing = jnp.broadcast_to(size, index.shape[:-1]).astype(jnp.int32)
    start = (index, missing, jnp.zeros((*index.shape[:-1], n_players), dtype=bool))
    return jax.lax.fori_loop(0, n_players, visit, start)[2]
