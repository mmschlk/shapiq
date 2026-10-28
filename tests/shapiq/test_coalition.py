"""Contracts for boolean coalition packing and unpacking."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from shapiq.coalition.coalition import pack_bool_coalition, unpack_bool_coalition


@pytest.mark.parametrize("n_players", [0, 1, 7, 8, 9, 16])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1), (0,), (2, 0)])
def test_bool_coalition_round_trip(n_players, batch_shape):
    coalition = jnp.broadcast_to(jnp.arange(n_players) % 2 == 0, (*batch_shape, n_players))
    packed = pack_bool_coalition(coalition)
    assert packed.shape == (*batch_shape, (n_players + 7) // 8)
    assert packed.dtype == jnp.uint8
    unpacked = unpack_bool_coalition(packed, n_players)
    assert unpacked.dtype == jnp.bool_
    assert bool(jnp.array_equal(unpacked, coalition))


@pytest.mark.parametrize("coalition", [jnp.ones((9,)), jnp.ones((), dtype=bool), [True, False]])
def test_pack_rejects_invalid_coalitions(coalition):
    with pytest.raises(TypeCheckError):
        pack_bool_coalition(coalition)


@pytest.mark.parametrize("result_shape", [(), (1, 2), (2, 1), (2, 3)])
def test_pack_checks_local_batch_and_width_relationships(monkeypatch, result_shape):
    monkeypatch.setattr(
        jnp, "packbits", lambda *args, **kwargs: jnp.zeros(result_shape, dtype=jnp.uint8)
    )
    with pytest.raises(TypeCheckError, match="return value"):
        pack_bool_coalition(jnp.ones((2, 9), dtype=bool))


@pytest.mark.parametrize(
    ("packed", "n_players"),
    [
        (jnp.ones((2,)), 9),
        (jnp.ones((2,), dtype=jnp.uint16), 9),
        (jnp.array(1, dtype=jnp.uint8), 9),
        (jnp.ones((1,), dtype=jnp.uint8), 9),
        (jnp.ones((3,), dtype=jnp.uint8), 9),
        (jnp.ones((1,), dtype=jnp.uint8), 0),
    ],
    ids=["float", "uint16", "scalar", "too-short", "too-long", "zero-players"],
)
def test_unpack_rejects_invalid_packed_coalitions(packed, n_players):
    with pytest.raises(TypeCheckError):
        unpack_bool_coalition(packed, n_players)


def test_coalition_conversions_support_jit():
    coalition = jnp.ones((2, 9), dtype=bool)
    packed = jax.jit(pack_bool_coalition)(coalition)
    unpacked = jax.jit(unpack_bool_coalition, static_argnames="n_players")(packed, 9)
    assert bool(jnp.array_equal(unpacked, coalition))
