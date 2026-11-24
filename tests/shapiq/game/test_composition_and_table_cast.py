"""Tests for ndlike 0.8.0 batch composition, JIT structural rebuilds, and TableGame casting."""

import jax
import jax.numpy as jnp
import pytest
from flextype import CastUnavailable, try_cast
from ndlike import jax_functions as jf
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.game import AffineGame, MappedGame, TableGame
from shapiq.game.discretederivative import DiscreteDerivative, subset_patterns
from shapiq.game.exact import ExactComputer
from shapiq.game.finite import FinitePlayerGame
from shapiq.game.fourier import FourierGame
from shapiq.game.moebius import AdditiveMoebiusGame, MoebiusGame
from shapiq.players import FinitePlayers, LatentPlayers


def _table(values: list[float] | list[list[float]], players: FinitePlayers) -> TableGame:
    return TableGame(jnp.array(values, dtype=jnp.float32), players)


def test_table_game_from_game_fast_path_and_recursive_leaf_check(monkeypatch):
    p = FinitePlayers(2)
    all_sets = subset_patterns(2)
    t1 = _table([[0.0, 1.0, 2.0, 4.0], [1.0, 2.0, 3.0, 5.0]], p)
    t2 = _table([0.5, 1.5, 2.5, 3.5], p)  # scalar batch broadcast with t1 (2,)

    # Already a TableGame returns itself at cost 0.
    assert TableGame.from_game(t1) is t1

    # Pure ArithmeticGame trees with TableGame leaves cast without calling eval().
    def fail_eval(*args, **kwargs):
        msg = "Fast ArithmeticGame -> TableGame cast must not call eval()"
        raise AssertionError(msg)

    aff = 2.0 * t1 + 3.0 * t2 - 1.0
    nested = (t1 + t2) * (t1 / 2.0)
    mapped = MappedGame(JaxBroadcastTuple((t1, t2)), lambda vs: vs[0] * 2.0 + vs[1])

    with monkeypatch.context() as mp:
        mp.setattr(FinitePlayerGame, "eval", fail_eval)
        for expr in (aff, nested, mapped):
            casted = TableGame.from_game(expr)
            assert isinstance(casted, TableGame)
            assert casted.shape == (2,)
            assert casted.players is t1.players
            # ExactComputer.from_game also uses the fast path.
            comp = ExactComputer.from_game(expr)
            assert isinstance(comp.table, TableGame)
            assert jnp.array_equal(comp.table.values, casted.values)

    for expr in (aff, nested, mapped):
        casted = TableGame.from_game(expr)
        assert jnp.allclose(casted(all_sets[:, None]), expr(all_sets[:, None]))

    # Mixed tree with a non-TableGame leaf declines the cost<=1.0 fast path and falls back to full eval.
    mg = MoebiusGame(
        values=jnp.array([1.0, 2.0]),
        interactions=jnp.array([[0, 0], [1, 0]], dtype=bool),
        players=p,
        additivity=1,
    )
    mixed = t1 + mg
    assert try_cast(mixed, TableGame, max_cost=1.0) is None
    mixed_table = TableGame.from_game(mixed)
    assert isinstance(mixed_table, TableGame)
    assert jnp.allclose(mixed_table(all_sets[:, None]), mixed(all_sets[:, None]))

    # Non-floating values cannot cast to TableGame.
    int_mapped = MappedGame(
        JaxBroadcastTuple((t1,)),
        lambda vs: vs[0].astype(jnp.int32),
        output_dtype=jnp.int32,
    )
    with pytest.raises(CastUnavailable):
        TableGame.from_game(int_mapped)


def test_table_game_at_set_casts_arithmetic_and_preserves_compact_players_under_jit():
    p = FinitePlayers(2)
    g = _table([[0.0, 1.0, 2.0, 4.0], [1.0, 3.0, 5.0, 7.0]], p)

    # Eager and JIT replacement with an arithmetic expression.
    updated = g.at[1].set(g[1] * 2.0)
    assert isinstance(updated, TableGame)
    assert jnp.allclose(updated.values[0], g.values[0])
    assert jnp.allclose(updated.values[1], g.values[1] * 2.0)
    assert jnp.array_equal(updated.players.ids, jnp.broadcast_to(p.ids, (2, 2)))

    # Direct replacement sharing the exact same scalar players object stays compact!
    g_same_p = _table([10.0, 20.0, 30.0, 40.0], p)
    updated_compact = g.at[1].set(g_same_p)
    assert updated_compact.players is p
    assert updated_compact.players.shape == ()

    jitted = jax.jit(lambda x: x.at[1].set(x[1] * 2.0))(g)
    assert isinstance(jitted, TableGame)
    assert jnp.allclose(jitted.values, updated.values)

    # Replacement with a MoebiusGame works via general TableGame cast.
    mg = MoebiusGame(
        values=jnp.array([2.0, 3.0]),
        interactions=jnp.array([[0, 0], [0, 1]], dtype=bool),
        players=p,
        additivity=1,
    )
    with_mg = g.at[0].set(mg)
    assert jnp.allclose(with_mg.values[0], jnp.array([2.0, 2.0, 5.0, 5.0]))
    assert with_mg.players is p

    # Mismatched LatentPlayers catalog or latent_n_players is rejected.
    cat1 = LatentPlayers(["a", "b"])
    cat2 = LatentPlayers(["a", "b"])
    g_cat1 = _table([[0.0, 1.0, 2.0, 3.0]], FinitePlayers(cat1))
    g_cat2 = _table([0.0, 1.0, 2.0, 3.0], FinitePlayers(cat2))
    with pytest.raises(ValueError, match="catalog"):
        g_cat1.at[0].set(g_cat2)

    g_lat3 = _table([0.0, 1.0, 2.0, 3.0], FinitePlayers(2, latent_n_players=3))
    with pytest.raises(ValueError, match="latent_n_players"):
        g.at[0].set(g_lat3)


def test_finite_players_and_table_game_stack_concat_jit_and_compact_broadcast():
    p = FinitePlayers(2)
    g1 = _table([0.0, 1.0, 2.0, 3.0], p)
    g2 = _table([1.0, 2.0, 3.0, 4.0], p)

    stacked = jf.jax_stack([g1, g2])
    assert isinstance(stacked, TableGame)
    assert stacked.shape == (2,)
    assert stacked.players is p
    assert stacked.players.shape == ()

    jitted_stack = jax.jit(lambda a, b: jf.jax_stack([a, b]))(g1, g2)
    assert jnp.allclose(jitted_stack.values, stacked.values)
    assert jnp.array_equal(jitted_stack.players.ids, jnp.broadcast_to(p.ids, (2, 2)))

    # Distinct FinitePlayers materialize cleanly under JIT too.
    p_a = FinitePlayers(jnp.array([0, 1]), latent_n_players=3)
    p_b = FinitePlayers(jnp.array([1, 2]), latent_n_players=3)
    ga = _table([0.0, 1.0, 2.0, 3.0], p_a)
    gb = _table([1.0, 2.0, 3.0, 4.0], p_b)
    jitted_diff = jax.jit(lambda a, b: jf.jax_stack([a, b]))(ga, gb)
    assert jitted_diff.players.shape == (2,)
    assert jnp.array_equal(jitted_diff.players.ids, jnp.array([[0, 1], [1, 2]]))

    # Mismatched catalog or latent_n_players rejected on stack/concat.
    p_cat1 = FinitePlayers(LatentPlayers(["a", "b"]))
    p_cat2 = FinitePlayers(LatentPlayers(["a", "b"]))
    with pytest.raises(ValueError, match="Static field 'players'"):
        jf.jax_stack([p_cat1, p_cat2])
    with pytest.raises(ValueError, match="Static field 'latent_n_players'"):
        jf.jax_stack([p, FinitePlayers(2, latent_n_players=3)])


def test_basis_game_composition_lcca_additivity_and_assignment():
    p = FinitePlayers(2)
    inter = jnp.array([[0, 0], [1, 0], [1, 1]], dtype=bool)
    inter1 = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=bool)

    mg2 = MoebiusGame(
        values=jnp.array([1.0, 2.0, 3.0]), interactions=inter, players=p, additivity=2
    )
    mg1 = MoebiusGame(
        values=jnp.array([1.0, 2.0, 0.0]), interactions=inter1, players=p, additivity=1
    )
    am1 = AdditiveMoebiusGame(values=jnp.array([1.0, 2.0, 3.0]), interactions=inter1, players=p)
    am0 = AdditiveMoebiusGame(
        values=jnp.array([5.0, 0.0, 0.0]),
        interactions=jnp.zeros((3, 2), dtype=bool),
        players=p,
        additivity=0,
    )
    fg = FourierGame(values=jnp.array([1.0, 2.0, 3.0]), interactions=inter, players=p, additivity=2)

    # Order-independent additivity = max(...) on MoebiusGame stack.
    s12 = jf.jax_stack([mg1, mg2])
    s21 = jf.jax_stack([mg2, mg1])
    assert type(s12) is MoebiusGame
    assert type(s21) is MoebiusGame
    assert s12.additivity == 2
    assert s21.additivity == 2
    assert s12.players is p

    # LCCA: MoebiusGame + AdditiveMoebiusGame -> MoebiusGame; AdditiveMoebiusGame(0) + (1) -> AdditiveMoebiusGame(1).
    mixed_moebius = jf.jax_stack([mg2, am1])
    assert type(mixed_moebius) is MoebiusGame
    assert mixed_moebius.additivity == 2

    mixed_add = jf.jax_stack([am0, am1])
    assert type(mixed_add) is AdditiveMoebiusGame
    assert mixed_add.additivity == 1

    # Sibling BasisGame types (MoebiusGame + FourierGame) are rejected.
    with pytest.raises((TypeError, ValueError)):
        jf.jax_stack([mg2, fg])

    # Compact interactions & players preserved when identical by identity.
    s_same = jf.jax_stack([mg2, mg2])
    assert s_same.players is p
    assert s_same.interactions is inter

    # Assignment enforces basis type and additivity bound (eager and JIT).
    mg1_batch = jf.jax_stack([mg1, mg1])
    with pytest.raises(ValueError, match="additivity"):
        mg1_batch.at[0].set(mg2)
    with pytest.raises(ValueError, match="additivity"):
        jax.jit(lambda dst, src: dst.at[0].set(src))(mg1_batch, mg2)
    with pytest.raises(TypeError, match="Cannot assign"):
        mg1_batch.at[0].set(fg)


def test_discrete_derivative_and_arithmetic_game_composition():
    p = FinitePlayers(3)
    t = _table([0.0, 1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 10.0], p)
    d1 = DiscreteDerivative(t, jnp.array([True, False, False]), order=1)
    d1_b = DiscreteDerivative(t, jnp.array([False, True, False]), order=1)
    d2 = DiscreteDerivative(t, jnp.array([True, True, False]), order=2)

    stacked_d_eager = jf.jax_stack([d1, d1_b])
    assert isinstance(stacked_d_eager, DiscreteDerivative)
    assert stacked_d_eager.order == 1
    assert stacked_d_eager.game is t

    stacked_d = jax.jit(lambda a, b: jf.jax_stack([a, b]))(d1, d1_b)
    assert isinstance(stacked_d, DiscreteDerivative)
    assert stacked_d.order == 1
    assert jnp.array_equal(stacked_d.interaction, stacked_d_eager.interaction)

    updated_d = jax.jit(lambda dst, src: dst.at[0].set(src))(stacked_d, d1_b)
    assert jnp.array_equal(updated_d.interaction[0], d1_b.interaction)

    with pytest.raises(ValueError, match="order"):
        stacked_d.at[0].set(d2)
    with pytest.raises(ValueError, match="Static field 'order'"):
        jf.jax_stack([d1, d2])

    # ArithmeticGame LCCA: _AdditiveAffineGame + _KAdditiveAffineGame -> _KAdditiveAffineGame
    # (and inner AdditiveMoebiusGame + MoebiusGame -> MoebiusGame).
    am = AdditiveMoebiusGame(
        values=jnp.array([1.0, 2.0, 3.0]),
        interactions=jnp.eye(3, dtype=bool),
        players=p,
    )
    mg2 = MoebiusGame(
        values=jnp.array([1.0, 2.0, 3.0]),
        interactions=jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
        players=p,
        additivity=2,
    )
    a_add = 2.0 * am
    a_k = 2.0 * mg2
    stacked_aff = jf.jax_stack([a_add, a_k])
    assert isinstance(stacked_aff, AffineGame)
    assert stacked_aff.shape == (2,)
    assert stacked_aff.additivity == 2

    # Heterogeneous child game types (AdditiveMoebiusGame + TableGame) are rejected.
    with pytest.raises(TypeError, match="common AxisProtected ancestor"):
        jf.jax_stack([a_add, 2.0 * t])

    # MappedGame static metadata (fn equality, _output_weak_type combination).
    fn1 = lambda vs: vs[0] + 1.0  # noqa: E731
    fn2 = lambda vs: vs[0] + 2.0  # noqa: E731
    m1 = MappedGame(JaxBroadcastTuple((t,)), fn1)
    m2 = MappedGame(JaxBroadcastTuple((t,)), fn2)
    m_same = jf.jax_stack([m1, m1])
    assert isinstance(m_same, MappedGame)
    with pytest.raises(ValueError, match="Static field 'fn'"):
        jf.jax_stack([m1, m2])

    # _IntPowerGame exponent mismatch & _ScalarDivGame reciprocal mismatch rejected.
    with pytest.raises(ValueError, match="Static field 'exponent'"):
        jf.jax_stack([t**2, t**3])
    with pytest.raises(ValueError, match="Static field 'reciprocal'"):
        jf.jax_stack([t / 2.0, 2.0 / t])
