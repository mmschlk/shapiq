"""Execute the sketchbook itself and exercise its runtime shape contracts."""

import json
import sys
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import FinitePlayerGame, PackedFinitePlayerGame


@pytest.fixture(scope="module")
def sketchbook(request: pytest.FixtureRequest) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples" / "game_api.ipynb"
    notebook = json.loads(path.read_text())
    module = ModuleType(f"{__name__}.sketchbook")
    sys.modules[module.__name__] = module
    # A normal fixture stays recognizable after runtime-checking wrappers.
    request.addfinalizer(lambda: sys.modules.pop(module.__name__))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = compile(
                "".join(cell["source"]),
                f"{path}:{cell['id']}",
                "exec",
                dont_inherit=True,
            )
            exec(code, module.__dict__)
    return module


def test_sketchbook_field_contracts_are_runtime_types(sketchbook):
    assert not isinstance(sketchbook.FinitePlayers.__annotations__["ids"], str)
    assert not isinstance(sketchbook.TableGame.__annotations__["values"], str)
    assert not isinstance(sketchbook.TableGame.__annotations__["_players"], str)


@pytest.mark.parametrize(
    "ids",
    [jnp.arange(3, dtype=jnp.float32), jnp.array(3)],
    ids=["wrong-dtype", "missing-player-axis"],
)
def test_players_reject_invalid_storage(sketchbook, ids):
    with pytest.raises(TypeCheckError):
        sketchbook.FinitePlayers(ids)


@pytest.mark.parametrize(
    "values",
    [jnp.arange(8)[None, :], jnp.zeros((1, 7)), jnp.array(1.0)],
    ids=["wrong-dtype", "wrong-coalition-count", "missing-coalition-axis"],
)
def test_table_rejects_invalid_storage(sketchbook, values):
    players = sketchbook.FinitePlayers(jnp.arange(3))
    with pytest.raises(TypeCheckError):
        sketchbook.TableGame(values, players)


@pytest.mark.parametrize("player_batch", [(2,), (1, 1)])
def test_players_cannot_enlarge_game_batch(sketchbook, player_batch):
    players = sketchbook.FinitePlayers(jnp.zeros((*player_batch, 3), dtype=int))
    with pytest.raises(TypeCheckError):
        sketchbook.TableGame(jnp.zeros((1, 8)), players)


@pytest.mark.parametrize("player_batch", [(), (3,), (2, 1), (2, 3)])
def test_players_broadcast_into_game_batch(sketchbook, player_batch):
    players = sketchbook.FinitePlayers(jnp.zeros((*player_batch, 3), dtype=int))
    game = sketchbook.TableGame(jnp.zeros((2, 3, 8)), players)
    assert game.players.shape == player_batch
    assert game(jnp.ones((3,), dtype=bool)).shape == (2, 3)


def test_coalition_batch_can_enlarge_evaluation_batch(sketchbook):
    game = sketchbook.TableGame(
        jnp.arange(8, dtype=jnp.float32)[None, :], sketchbook.FinitePlayers(jnp.arange(3))
    )
    coalitions = jnp.ones((5, 1, 3), dtype=bool)
    assert game.is_valid(coalitions)
    values = game(coalitions)
    assert values.shape == (5, 1)
    assert bool(jnp.all(values == 7))


def test_shared_players_need_the_game_batch_when_validating_coalitions(sketchbook):
    players = sketchbook.FinitePlayers(3)
    game = sketchbook.TableGame(jnp.zeros((2, 8)), players)
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
def test_invalid_coalitions_fail_checked_evaluation(sketchbook, coalition, error):
    game = sketchbook.TableGame(jnp.zeros((2, 8)), sketchbook.FinitePlayers(jnp.arange(3)))
    assert not game.is_valid(coalition)
    with pytest.raises(error):
        game(coalition)


def test_zero_player_table_has_exactly_one_coalition(sketchbook):
    players = sketchbook.FinitePlayers(jnp.arange(0))
    game = sketchbook.TableGame(jnp.array([7.0]), players)
    assert game.shape == ()
    assert float(game(jnp.zeros((0,), dtype=bool))) == 7.0
    with pytest.raises(TypeCheckError):
        sketchbook.TableGame(jnp.zeros((2,)), players)


def test_shape_contracts_work_with_jit_and_game_pytrees(sketchbook):
    game = sketchbook.TableGame(
        jnp.arange(16, dtype=jnp.float32).reshape((2, 8)),
        sketchbook.FinitePlayers(jnp.arange(3)),
    )
    coalitions = jnp.array([[0, 0, 0], [1, 1, 1]], dtype=bool)
    evaluate = jax.jit(lambda candidate, selections: candidate(selections))
    values = evaluate(game, coalitions)
    assert bool(jnp.allclose(values, jnp.array([0.0, 15.0])))


def test_return_contract_rejects_wrong_evaluation_batch(sketchbook, monkeypatch):
    game = sketchbook.TableGame(jnp.zeros((2, 8)), sketchbook.FinitePlayers(jnp.arange(3)))
    monkeypatch.setattr(
        sketchbook.jnp, "take_along_axis", lambda *args, **kwargs: jnp.zeros((1, 1))
    )
    with pytest.raises(TypeCheckError, match="return value"):
        game(jnp.zeros((3,), dtype=bool))


def test_table_implements_only_the_packed_native_hook(sketchbook):
    assert issubclass(sketchbook.TableGame, FinitePlayerGame)
    assert issubclass(sketchbook.TableGame, PackedFinitePlayerGame)
    assert "eval" not in sketchbook.TableGame.__dict__
    assert "_eval_bool" not in sketchbook.TableGame.__dict__
    assert sketchbook.TableGame._eval_bool is PackedFinitePlayerGame._eval_bool
    assert "_eval_packed" in sketchbook.TableGame.__dict__


@pytest.mark.parametrize("n_players", [0, 1, 3, 8, 9, 16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("x64", [False, True])
def test_packed_table_indexes_bytes_directly(sketchbook, monkeypatch, n_players, compiled, x64):
    with jax.enable_x64(new_val=x64):
        game = sketchbook.TableGame(
            jnp.arange(1 << n_players, dtype=jnp.float32), sketchbook.FinitePlayers(n_players)
        )
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
