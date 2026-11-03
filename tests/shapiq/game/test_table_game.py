"""Dense table games, runtime contracts, and compiled evaluation."""

from math import prod

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import FinitePlayerGame, PackedFinitePlayerGame, TableGame
from shapiq.players import FinitePlayers


def test_field_contracts_are_runtime_types():
    assert not isinstance(FinitePlayers.__annotations__["ids"], str)
    assert not isinstance(TableGame.__annotations__["values"], str)
    assert not isinstance(TableGame.__annotations__["players"], str)


@pytest.mark.parametrize(
    "ids",
    [jnp.arange(3, dtype=jnp.float32), jnp.array(3)],
    ids=["wrong-dtype", "missing-player-axis"],
)
def test_players_reject_invalid_storage(ids):
    with pytest.raises(TypeCheckError):
        FinitePlayers(ids)


@pytest.mark.parametrize(
    "values",
    [jnp.arange(8)[None, :], jnp.zeros((1, 7)), jnp.array(1.0)],
    ids=["wrong-dtype", "wrong-coalition-count", "missing-coalition-axis"],
)
def test_table_rejects_invalid_storage(values):
    players = FinitePlayers(jnp.arange(3))
    with pytest.raises(TypeCheckError):
        TableGame(values, players)


@pytest.mark.parametrize("player_batch", [(2,), (1, 1)])
def test_players_cannot_enlarge_game_batch(player_batch):
    players = FinitePlayers(jnp.zeros((*player_batch, 3), dtype=int))
    with pytest.raises(TypeCheckError):
        TableGame(jnp.zeros((1, 8)), players)


@pytest.mark.parametrize("player_batch", [(), (3,), (2, 1), (2, 3)])
def test_players_broadcast_into_game_batch(player_batch):
    players = FinitePlayers(jnp.zeros((*player_batch, 3), dtype=int))
    game = TableGame(jnp.zeros((2, 3, 8)), players)
    assert game.players.shape == player_batch
    assert game(jnp.ones((3,), dtype=bool)).shape == (2, 3)


@pytest.mark.parametrize("player_batch", [(), (1, 1), (3,), (2, 1), (2, 3)])
def test_table_batch_operations_keep_broadcast_players_aligned(player_batch):
    ids = jnp.arange(3 * prod(player_batch)).reshape((*player_batch, 3))
    players = FinitePlayers(ids, players=list(range(18)))
    values = jnp.arange(48, dtype=jnp.float32).reshape(2, 3, 8)
    game = TableGame(values, players)
    expanded_ids = jnp.broadcast_to(ids, (2, 3, 3))

    for result, expected_values, expected_ids in [
        (game[1], values[1], expanded_ids[1]),
        (game[:, 1], values[:, 1], expanded_ids[:, 1]),
        (game.reshape(6), values.reshape(6, 8), expanded_ids.reshape(6, 3)),
        (game.T, values.transpose(1, 0, 2), expanded_ids.transpose(1, 0, 2)),
    ]:
        assert bool(jnp.array_equal(result.values, expected_values))
        assert bool(
            jnp.array_equal(jnp.broadcast_to(result.players.ids, expected_ids.shape), expected_ids)
        )
        assert result.players.players is players.players
        assert result.players.latent_n_players == players.latent_n_players
        assert bool(jnp.array_equal(result(jnp.ones(3, dtype=bool)), expected_values[..., -1]))
        if players.size == 1:
            assert result.players.size == 1


def test_coalition_batch_can_enlarge_evaluation_batch():
    game = TableGame(jnp.arange(8, dtype=jnp.float32)[None, :], FinitePlayers(jnp.arange(3)))
    coalitions = jnp.ones((5, 1, 3), dtype=bool)
    assert game.is_valid(coalitions)
    values = game(coalitions)
    assert values.shape == (5, 1)
    assert bool(jnp.all(values == 7))


def test_shared_players_need_the_game_batch_when_validating_coalitions():
    players = FinitePlayers(3)
    game = TableGame(jnp.zeros((2, 8)), players)
    coalition = jnp.ones((3, 3), dtype=bool)
    assert players.is_valid(coalition)
    assert not game.is_valid(coalition)


@pytest.mark.parametrize(
    ("coalition", "error"),
    [
        (jnp.zeros((3,)), TypeCheckError),
        (jnp.zeros((4,), dtype=bool), TypeCheckError),
        (jnp.ones((), dtype=bool), TypeCheckError),
        (jnp.zeros((3, 3), dtype=bool), ValueError),
        ([True, False, False], TypeCheckError),
    ],
    ids=["wrong-dtype", "wrong-players", "missing-player-axis", "wrong-batch", "not-an-array"],
)
def test_invalid_coalitions_fail_evaluation(coalition, error):
    game = TableGame(jnp.zeros((2, 8)), FinitePlayers(jnp.arange(3)))
    assert not game.is_valid(coalition)
    with pytest.raises(error):
        game(coalition)


def test_zero_player_table_has_exactly_one_coalition():
    players = FinitePlayers(jnp.arange(0))
    game = TableGame(jnp.array([7.0]), players)
    assert game.shape == ()
    assert float(game(jnp.zeros((0,), dtype=bool))) == 7.0
    with pytest.raises(TypeCheckError):
        TableGame(jnp.zeros((2,)), players)


def test_shape_contracts_work_with_jit_and_game_pytrees():
    game = TableGame(
        jnp.arange(16, dtype=jnp.float32).reshape((2, 8)),
        FinitePlayers(jnp.arange(3)),
    )
    coalitions = jnp.array([[0, 0, 0], [1, 1, 1]], dtype=bool)
    evaluate = jax.jit(lambda candidate, selections: candidate(selections))
    values = evaluate(game, coalitions)
    assert bool(jnp.allclose(values, jnp.array([0.0, 15.0])))

    changed_game = TableGame(game.values + 100, game.players)
    assert bool(jnp.allclose(evaluate(changed_game, coalitions), values + 100))
    assert bool(jnp.allclose(evaluate(game, ~coalitions), jnp.array([7.0, 8.0])))


def test_return_contract_rejects_wrong_evaluation_batch(monkeypatch):
    game = TableGame(jnp.zeros((2, 8)), FinitePlayers(jnp.arange(3)))
    monkeypatch.setattr(jnp, "take_along_axis", lambda *args, **kwargs: jnp.zeros((1, 1)))
    # Exercise the patched lookup even if this shape has already been compiled.
    with jax.disable_jit(), pytest.raises(TypeCheckError, match="return value"):
        game(jnp.zeros((3,), dtype=bool))


@pytest.mark.parametrize("compiled", [False, True])
def test_packed_hook_checks_game_batch_in_return_shape(monkeypatch, compiled):
    game = TableGame(jnp.zeros((2, 8)), FinitePlayers(3))
    coalition = jnp.zeros((1, 1), dtype=jnp.uint8)
    # Shape (1,) agrees with the coalition batch but omits the game batch (2,).
    monkeypatch.setattr(jnp, "take_along_axis", lambda *args, **kwargs: jnp.zeros((1, 1)))
    TableGame._eval_packed.clear_cache()
    with jax.disable_jit(disable=not compiled), pytest.raises(TypeCheckError, match="return value"):
        game._eval_packed(coalition)


@pytest.mark.parametrize("compiled", [False, True])
def test_packed_hook_rejects_incompatible_batches(compiled):
    game = TableGame(jnp.zeros((2, 8)), FinitePlayers(3))
    coalition = jnp.zeros((3, 1), dtype=jnp.uint8)
    with jax.disable_jit(disable=not compiled), pytest.raises(TypeCheckError, match="batch"):
        game._eval_packed(coalition)


def test_table_implements_only_the_packed_native_hook():
    assert issubclass(TableGame, FinitePlayerGame)
    assert issubclass(TableGame, PackedFinitePlayerGame)
    assert "eval" not in TableGame.__dict__
    assert "_eval_bool" not in TableGame.__dict__
    assert TableGame._eval_bool is PackedFinitePlayerGame._eval_bool
    assert "_eval_packed" in TableGame.__dict__


@pytest.mark.parametrize("n_players", [0, 1, 3, 8, 9, 16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("x64", [False, True])
def test_packed_table_indexes_bytes_directly(monkeypatch, n_players, compiled, x64):
    with jax.enable_x64(new_val=x64):
        game = TableGame(jnp.arange(1 << n_players, dtype=jnp.float32), FinitePlayers(n_players))
        indices = jnp.array([0, 1, 7, 8, 255, 256, (1 << n_players) - 1], dtype=jnp.uint32)
        indices &= jnp.uint32((1 << n_players) - 1)
        coalition = ((indices[:, None] >> jnp.arange(n_players)) & 1).astype(bool)
        packed = pack_bool_coalition(coalition)
        if n_players % 8:
            padding = jnp.uint8(255 ^ ((1 << (n_players % 8)) - 1))
            packed = packed.at[..., -1].set(packed[..., -1] | padding)

        def unexpected_unpack(*args, **kwargs):
            msg = "Packed table indexing must not unpack coalition membership"
            raise AssertionError(msg)

        monkeypatch.setattr(jnp, "unpackbits", unexpected_unpack)

        def evaluate(g, c):
            return g(c)

        if compiled:
            evaluate = jax.jit(evaluate)
        assert bool(jnp.array_equal(evaluate(game, packed), indices.astype(jnp.float32)))
