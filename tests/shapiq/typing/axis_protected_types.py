"""Static contracts for ndlike batch operations; checked by the ty lint hook."""

from dataclasses import dataclass
from typing import Any, assert_type, override

import jax
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.game._arithmetic_ops import (
    _AdditiveDivGame,
    _IntPowerGame,
    _KAdditiveDivGame,
    _KAdditivePowerGame,
    _ScalarDivGame,
)
from shapiq.game.arithmetic import (
    AffineGame,
    ArithmeticGame,
    MappedGame,
    ProductGame,
    _AdditiveAffineGame,
    _KAdditiveAffineGame,
    _KAdditiveProductGame,
)
from shapiq.game.discretederivative import DiscreteDerivative
from shapiq.game.exact import ExactComputer
from shapiq.game.finite import FinitePlayerGame
from shapiq.game.fourier import FourierGame
from shapiq.game.kadditive import AdditiveGame, KAdditiveGame
from shapiq.game.moebius import AdditiveMoebiusGame, MoebiusGame
from shapiq.game.table import TableGame
from shapiq.index import BanzhafIndex, ShapleyValueIndex
from shapiq.players import FinitePlayers
from shapiq.sampling._base import SizeWithReplacement
from shapiq.sampling._state import DenseSample, Sample
from shapiq.value.value import JaxValue


def batch_results(
    *,
    players: FinitePlayers[str],
    table: TableGame[str],
    exact: ExactComputer,
    derivative: DiscreteDerivative[str, jax.Array],
    fourier: FourierGame[str, jax.Array],
    moebius: MoebiusGame[str, jax.Array],
    additive: AdditiveMoebiusGame[str, jax.Array],
    affine: AffineGame[str, jax.Array],
    product: ProductGame[str, jax.Array],
    mapped: MappedGame[str, jax.Array],
    banzhaf: BanzhafIndex,
    shapley: ShapleyValueIndex,
    sampler: SizeWithReplacement,
    sample: DenseSample,
) -> None:
    assert_type(players[0], FinitePlayers[str])
    assert_type(players.reshape(-1), FinitePlayers[str])
    assert_type(next(iter(players)), FinitePlayers[str])
    assert_type(players.at[0].get(), FinitePlayers[str])
    assert_type(players.at[0].set(players[0]), FinitePlayers[str])
    assert_type(table[0].reshape(-1), TableGame[str])
    assert_type(exact[0], ExactComputer)
    assert_type(derivative[0], DiscreteDerivative[str, jax.Array])
    assert_type(fourier[0], FourierGame[str, jax.Array])
    assert_type(moebius[0], MoebiusGame[str, jax.Array, Any])
    assert_type(additive[0].reshape(-1), AdditiveMoebiusGame[str, jax.Array])
    assert_type(affine[0].reshape(-1), AffineGame[str, jax.Array])
    assert_type(product[0].reshape(-1), ProductGame[str, jax.Array])
    assert_type(mapped[0].reshape(-1), MappedGame[str, jax.Array])
    assert_type(banzhaf[0], BanzhafIndex)
    assert_type(shapley[0].reshape(-1), ShapleyValueIndex)
    assert_type(sampler[0], SizeWithReplacement)
    assert_type(sample[0].reshape(-1), DenseSample)
    assert_type(sample.at[0].set(sample[0]), DenseSample)


def accepts_subclasses(additive: AdditiveMoebiusGame[str, jax.Array], dense: DenseSample) -> None:
    game: MoebiusGame[str, jax.Array, Any] = additive
    sample: Sample[Any] = dense
    assert_type(game.players, FinitePlayers[str])
    assert_type(sample.coalitions, jax.Array)


def arithmetic_results(
    table: TableGame[str], affine: AffineGame[str, jax.Array], offset: jax.Array
) -> None:
    adjusted = 2 * (table + 1) - 3
    product = table * adjusted
    quotient = product / (table + 1)
    power = table**2

    assert_type(+table, TableGame[str])
    assert_type(+affine, AffineGame[str, jax.Array])
    assert_type(-table, AffineGame[str, JaxValue])
    assert_type(adjusted, AffineGame[str, JaxValue])
    assert_type(adjusted.reshape(1), AffineGame[str, JaxValue])
    assert_type(product, ArithmeticGame[str, JaxValue])
    assert_type(quotient, ArithmeticGame[str, JaxValue])
    assert_type(quotient.players, FinitePlayers[str])
    assert_type(quotient.eval(table.players.empty_coalition()), JaxValue)
    assert_type(power, ArithmeticGame[str, JaxValue])
    assert_type(table**2.5, ArithmeticGame[str, JaxValue])
    assert_type(table**affine, ArithmeticGame[str, JaxValue])
    assert_type(table**offset, ArithmeticGame[str, JaxValue])
    assert_type(2**table, ArithmeticGame[str, JaxValue])
    assert_type(offset**table, ArithmeticGame[str, JaxValue])
    assert_type(pow(table, 2), ArithmeticGame[str, JaxValue])
    assert_type(pow(table, 2, None), ArithmeticGame[str, JaxValue])
    assert_type(1 + table, AffineGame[str, JaxValue])
    assert_type(1 - table, AffineGame[str, JaxValue])
    assert_type(1 * table, ArithmeticGame[str, JaxValue])
    assert_type(1 / table, ArithmeticGame[str, JaxValue])
    assert_type(offset + table, AffineGame[str, JaxValue])
    assert_type(affine + table, AffineGame[str, JaxValue])


def arithmetic_operand_types(
    *,
    table: TableGame[str],
    moebius: MoebiusGame[str, jax.Array],
    additive: AdditiveMoebiusGame[str, jax.Array],
    weight: jax.Array,
) -> None:
    # Default GameT = FinitePlayerGame[P, Any] with explicit P, V checks player compatibility.
    default_affine = AffineGame[str, jax.Array](
        JaxBroadcastTuple((table, additive)), JaxBroadcastTuple((weight, weight))
    )
    assert_type(default_affine, AffineGame[str, jax.Array])
    assert_type(default_affine.games.members, tuple[FinitePlayerGame[str, Any], ...])
    assert_type(default_affine.players, FinitePlayers[str])
    assert_type(default_affine[0].reshape(-1), AffineGame[str, jax.Array])

    default_product = ProductGame[str, jax.Array](JaxBroadcastTuple((table, additive)))
    assert_type(default_product, ProductGame[str, jax.Array])
    assert_type(default_product.games.members, tuple[FinitePlayerGame[str, Any], ...])
    assert_type(default_product.players, FinitePlayers[str])

    default_mapped = MappedGame[str, jax.Array](JaxBroadcastTuple((table,)), lambda _vs: weight)
    assert_type(default_mapped, MappedGame[str, jax.Array])
    assert_type(default_mapped.players, FinitePlayers[str])
    assert_type(default_mapped.eval(table.players.empty_coalition()), jax.Array)

    # Nested rewriting wrappers type stored operands conservatively as FinitePlayerGame,
    # never promising inner AffineGame or original operand attributes after flattening/autocast.
    inner_affine = table + 2
    outer_affine = AffineGame(JaxBroadcastTuple((inner_affine,)), JaxBroadcastTuple((weight,)))
    assert_type(outer_affine.games.members, tuple[FinitePlayerGame[Any, Any], ...])
    _ = outer_affine.games.members[0].weights  # ty: ignore[missing-attribute]

    outer_typed = AffineGame[str, jax.Array](
        JaxBroadcastTuple((inner_affine,)), JaxBroadcastTuple((weight,))
    )
    assert_type(outer_typed, AffineGame[str, jax.Array])
    assert_type(outer_typed.games.members, tuple[FinitePlayerGame[str, Any], ...])
    assert_type(outer_typed.players, FinitePlayers[str])
    assert_type(outer_typed[0].reshape(-1), AffineGame[str, jax.Array])
    _ = outer_typed.games.members[0].weights  # ty: ignore[missing-attribute]

    # ProductGame, _ScalarDivGame, and _IntPowerGame can rewrite operands via zero-cost autocast.
    inferred_product = ProductGame(JaxBroadcastTuple((table,)))
    assert_type(inferred_product.games.members, tuple[FinitePlayerGame[Any, Any], ...])
    _ = inferred_product.games.members[0].values  # ty: ignore[missing-attribute]

    inferred_div = _ScalarDivGame(JaxBroadcastTuple((table,)), weight)
    assert_type(inferred_div.games.members, tuple[FinitePlayerGame[Any, Any], ...])
    _ = inferred_div.games.members[0].values  # ty: ignore[missing-attribute]

    inferred_pow = _IntPowerGame(JaxBroadcastTuple((table,)), 2)
    assert_type(inferred_pow.games.members, tuple[FinitePlayerGame[Any, Any], ...])
    _ = inferred_pow.games.members[0].values  # ty: ignore[missing-attribute]

    # Non-rewriting MappedGame and custom ArithmeticGame subclasses retain precise GameT.
    inferred_mapped = MappedGame(JaxBroadcastTuple((additive,)), lambda _vs: weight)
    assert_type(inferred_mapped.games.members, tuple[AdditiveMoebiusGame[str, jax.Array], ...])
    assert_type(inferred_mapped.eval(table.players.empty_coalition()), jax.Array)

    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class CustomArithmetic[
        P,
        V: JaxValue = JaxValue,
        GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
    ](ArithmeticGame[P, V, GameT, "CustomArithmetic[P, V, GameT]"]):
        @override
        def _compute(self, values: tuple[JaxValue, ...]) -> Any:
            return values[0]

    custom_arith = CustomArithmetic(JaxBroadcastTuple((additive,)))
    assert_type(custom_arith.games.members, tuple[AdditiveMoebiusGame[str, jax.Array], ...])

    # Dataclass subclasses of AffineGame preserve synthesized constructors.
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class CustomAffine(AffineGame[Any, Any]):
        extra: int = 0

    custom_affine = CustomAffine(JaxBroadcastTuple((table,)), JaxBroadcastTuple((weight,)), extra=1)
    assert_type(custom_affine, CustomAffine)
    assert_type(custom_affine.extra, int)

    # Private refinements carry narrowed KAdditiveGame / AdditiveGame children without casts,
    # and flattening affine refinements do not promise narrower operand subclasses.
    kadd_affine = _KAdditiveAffineGame[str, jax.Array](
        JaxBroadcastTuple((moebius,)), JaxBroadcastTuple((weight,))
    )
    assert_type(kadd_affine.games.members, tuple[KAdditiveGame[str, Any], ...])
    assert_type(kadd_affine.games.members[0].additivity, int)
    assert_type(kadd_affine.players, FinitePlayers[str])

    add_affine = _AdditiveAffineGame[str, jax.Array](
        JaxBroadcastTuple((additive,)), JaxBroadcastTuple((weight,))
    )
    assert_type(add_affine.games.members, tuple[AdditiveGame[str, Any], ...])
    assert_type(add_affine.games.members[0].contributions(), Any)
    assert_type(add_affine.players, FinitePlayers[str])

    inferred_add_affine = _AdditiveAffineGame(
        JaxBroadcastTuple((additive,)), JaxBroadcastTuple((weight,))
    )
    assert_type(inferred_add_affine.games.members, tuple[AdditiveGame[Any, Any], ...])
    assert_type(inferred_add_affine.games.members[0].contributions(), Any)
    _ = inferred_add_affine.games.members[0].interactions  # ty: ignore[missing-attribute]

    kadd_product = _KAdditiveProductGame[str, jax.Array](JaxBroadcastTuple((moebius,)))
    assert_type(kadd_product.games.members, tuple[KAdditiveGame[str, Any], ...])
    assert_type(kadd_product.games.members[0].additivity, int)

    kadd_div = _KAdditiveDivGame[str, jax.Array](JaxBroadcastTuple((moebius,)), weight)
    assert_type(kadd_div.games.members, tuple[KAdditiveGame[str, Any], ...])
    assert_type(kadd_div.games.members[0].additivity, int)

    add_div = _AdditiveDivGame[str, jax.Array](JaxBroadcastTuple((additive,)), weight)
    assert_type(add_div.games.members, tuple[AdditiveGame[str, Any], ...])
    assert_type(add_div.games.members[0].contributions(), Any)

    kadd_power = _KAdditivePowerGame[str, jax.Array](JaxBroadcastTuple((moebius,)), 2)
    assert_type(kadd_power.games.members, tuple[KAdditiveGame[str, Any], ...])
    assert_type(kadd_power.games.members[0].additivity, int)

    # Explicit misleading operand specialization on rewriting wrappers must be rejected.
    AffineGame[str, jax.Array, AdditiveMoebiusGame[str, jax.Array]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((additive,)), JaxBroadcastTuple((weight,))
    )
    ProductGame[str, jax.Array, MoebiusGame[str, jax.Array]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((moebius,))
    )
    _KAdditiveAffineGame[str, jax.Array, MoebiusGame[str, jax.Array]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((moebius,)), JaxBroadcastTuple((weight,))
    )
    _AdditiveAffineGame[str, jax.Array, AdditiveMoebiusGame[str, jax.Array]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((additive,)), JaxBroadcastTuple((weight,))
    )
    _ScalarDivGame[str, jax.Array, TableGame[str]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((table,)), weight
    )
    _IntPowerGame[str, jax.Array, TableGame[str]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((table,)), 2
    )

    # Invalid construction must be rejected statically (verified with unused-ignore-comment).
    AffineGame(JaxBroadcastTuple((weight,)), JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    AffineGame[int, jax.Array](JaxBroadcastTuple((table,)), JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    ProductGame(JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    MappedGame(JaxBroadcastTuple((weight,)), lambda _vs: weight)  # ty: ignore[invalid-argument-type]
    ProductGame[int, jax.Array](JaxBroadcastTuple((table,)))  # ty: ignore[invalid-argument-type]
    _KAdditiveAffineGame(JaxBroadcastTuple((table,)), JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    _KAdditiveProductGame(JaxBroadcastTuple((table,)))  # ty: ignore[invalid-argument-type]
    _KAdditiveDivGame(JaxBroadcastTuple((table,)), weight)  # ty: ignore[invalid-argument-type]
    _KAdditivePowerGame(JaxBroadcastTuple((table,)), 2)  # ty: ignore[invalid-argument-type]
    _AdditiveAffineGame(JaxBroadcastTuple((moebius,)), JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    _AdditiveDivGame(JaxBroadcastTuple((moebius,)), weight)  # ty: ignore[invalid-argument-type]
