"""Static contracts for ndlike batch operations; checked by the ty lint hook."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, assert_type, override

import jax
from ndlike.protected_axis.jax import JaxAxisProtected
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.game import (
    AffineGame,
    ArithmeticGame,
    DenseCoalitionGame,
    DenseFourierGame,
    DenseMoebiusGame,
    DiscreteDerivative,
    FinitePlayerGame,
    MappedGame,
    ProductGame,
    ReductionGame,
    SparseCoalitionGame,
    SparseFourierGame,
    SparseMoebiusGame,
    SumReductionGame,
)
from shapiq.game._arithmetic_ops import _IntPowerGame, _ScalarDivGame
from shapiq.index import BanzhafIndex, ExactComputer, ShapleyValueIndex
from shapiq.players import FinitePlayers
from shapiq.sampling._base import SizeWithReplacement
from shapiq.sampling._state import DenseSample, Sample
from shapiq.value.value import JaxValue

if TYPE_CHECKING:
    import numpy as np

    from shapiq.game import Game


def batch_results(
    *,
    players: FinitePlayers[str],
    dense_coalition: DenseCoalitionGame[str],
    dense_moebius: DenseMoebiusGame[str],
    dense_fourier: DenseFourierGame[str],
    exact: ExactComputer,
    derivative: DiscreteDerivative[str, jax.Array],
    sparse_coalition: SparseCoalitionGame[str, jax.Array],
    sparse_fourier: SparseFourierGame[str, jax.Array],
    sparse_moebius: SparseMoebiusGame[str, jax.Array],
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
    assert_type(dense_coalition[0].reshape(-1), DenseCoalitionGame[str])
    assert_type(dense_moebius[0].reshape(-1), DenseMoebiusGame[str])
    assert_type(dense_fourier[0].reshape(-1), DenseFourierGame[str])
    assert_type(exact[0], ExactComputer)
    assert_type(derivative[0], DiscreteDerivative[str, jax.Array])
    assert_type(sparse_coalition[0], SparseCoalitionGame[str, jax.Array])
    assert_type(sparse_fourier[0], SparseFourierGame[str, jax.Array])
    assert_type(sparse_moebius[0].reshape(-1), SparseMoebiusGame[str, jax.Array])
    assert_type(affine[0].reshape(-1), AffineGame[str, jax.Array])
    assert_type(product[0].reshape(-1), ProductGame[str, jax.Array])
    assert_type(mapped[0].reshape(-1), MappedGame[str, jax.Array])
    assert_type(banzhaf[0], BanzhafIndex)
    assert_type(shapley[0].reshape(-1), ShapleyValueIndex)
    assert_type(sampler[0], SizeWithReplacement)
    assert_type(sample[0].reshape(-1), DenseSample)
    assert_type(sample.at[0].set(sample[0]), DenseSample)


def accepts_subclasses(sparse: SparseMoebiusGame[str, jax.Array], dense: DenseSample) -> None:
    game: FinitePlayerGame[str, JaxValue] = sparse
    sample: Sample[Any] = dense
    assert_type(game.players, FinitePlayers[str])
    assert_type(sample.coalitions, jax.Array)


def arithmetic_results(
    table: DenseCoalitionGame[str], affine: AffineGame[str, jax.Array], offset: jax.Array
) -> None:
    adjusted = 2 * (table + 1) - 3
    product = table * adjusted
    quotient = product / (table + 1)
    power = table**2

    assert_type(+table, DenseCoalitionGame[str])
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
    table: DenseCoalitionGame[str],
    moebius: SparseMoebiusGame[str, jax.Array],
    weight: jax.Array,
) -> None:
    # Default GameT = FinitePlayerGame[P, Any] with explicit P, V checks player compatibility.
    default_affine = AffineGame[str, jax.Array](
        JaxBroadcastTuple((table, moebius)), JaxBroadcastTuple((weight, weight))
    )
    assert_type(default_affine, AffineGame[str, jax.Array])
    assert_type(default_affine.games.members, tuple[FinitePlayerGame[str, Any], ...])
    assert_type(default_affine.players, FinitePlayers[str])
    assert_type(default_affine[0].reshape(-1), AffineGame[str, jax.Array])

    default_product = ProductGame[str, jax.Array](JaxBroadcastTuple((table, moebius)))
    assert_type(default_product, ProductGame[str, jax.Array])
    assert_type(default_product.games.members, tuple[FinitePlayerGame[str, Any], ...])
    assert_type(default_product.players, FinitePlayers[str])

    default_mapped = MappedGame[str, jax.Array](JaxBroadcastTuple((table,)), lambda _vs: weight)
    assert_type(default_mapped, MappedGame[str, jax.Array])
    assert_type(default_mapped.players, FinitePlayers[str])
    assert_type(default_mapped.eval(table.players.empty_coalition()), jax.Array)

    # Nested rewriting wrappers type stored operands conservatively as FinitePlayerGame.
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
    inferred_mapped = MappedGame(JaxBroadcastTuple((moebius,)), lambda _vs: weight)
    assert_type(inferred_mapped.games.members, tuple[SparseMoebiusGame[str, jax.Array], ...])
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

    custom_arith = CustomArithmetic(JaxBroadcastTuple((moebius,)))
    assert_type(custom_arith.games.members, tuple[SparseMoebiusGame[str, jax.Array], ...])

    # Dataclass subclasses of AffineGame preserve synthesized constructors.
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class CustomAffine(AffineGame[Any, Any]):
        extra: int = 0

    custom_affine = CustomAffine(JaxBroadcastTuple((table,)), JaxBroadcastTuple((weight,)), extra=1)
    assert_type(custom_affine, CustomAffine)
    assert_type(custom_affine.extra, int)

    # Explicit misleading operand specialization on rewriting wrappers must be rejected.
    AffineGame[str, jax.Array, SparseMoebiusGame[str, jax.Array]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((moebius,)), JaxBroadcastTuple((weight,))
    )
    ProductGame[str, jax.Array, SparseMoebiusGame[str, jax.Array]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((moebius,))
    )
    _ScalarDivGame[str, jax.Array, DenseCoalitionGame[str]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((table,)), weight
    )
    _IntPowerGame[str, jax.Array, DenseCoalitionGame[str]](  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((table,)), 2
    )

    # Invalid construction must be rejected statically (verified with unused-ignore-comment).
    AffineGame(JaxBroadcastTuple((weight,)), JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    AffineGame[int, jax.Array](JaxBroadcastTuple((table,)), JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    ProductGame(JaxBroadcastTuple((weight,)))  # ty: ignore[invalid-argument-type]
    MappedGame(JaxBroadcastTuple((weight,)), lambda _vs: weight)  # ty: ignore[invalid-argument-type]
    ProductGame[int, jax.Array](JaxBroadcastTuple((table,)))  # ty: ignore[invalid-argument-type]


def reduction_results(
    *,
    table: DenseCoalitionGame[str],
    moebius: SparseMoebiusGame[str, jax.Array],
    affine: AffineGame[str, jax.Array],
    weight: jax.Array,
) -> None:
    sum_table = table.sum()
    assert_type(sum_table, SumReductionGame[str, jax.Array])
    assert_type(sum_table[0].reshape(-1), SumReductionGame[str, jax.Array])
    assert_type(sum_table.players, FinitePlayers[str])
    assert_type(sum_table.eval(table.players.empty_coalition()), jax.Array)
    assert_type(sum_table.full_value(), jax.Array)
    assert_type(sum_table.contributions(), jax.Array)

    sum_affine = affine.sum(axis=0, keepdims=True)
    assert_type(sum_affine, SumReductionGame[str, jax.Array])
    assert_type(sum_affine.eval(table.players.empty_coalition()), jax.Array)

    sum_moebius = moebius.sum()
    assert_type(sum_moebius, SumReductionGame[str, JaxValue])

    @dataclass(frozen=True)
    class StructuredVal(JaxAxisProtected["StructuredVal"]):
        values: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    def _check_structured(
        structured_game: FinitePlayerGame[str, StructuredVal],
        game_valued_game: FinitePlayerGame[str, DenseCoalitionGame[int]],
    ) -> None:
        sum_struct = structured_game.sum()
        assert_type(sum_struct, SumReductionGame[str, StructuredVal])
        assert_type(sum_struct.eval(table.players.empty_coalition()), StructuredVal)
        assert_type(SumReductionGame(structured_game), SumReductionGame[str, StructuredVal])

        sum_game_valued = game_valued_game.sum()
        assert_type(sum_game_valued, SumReductionGame[str, FinitePlayerGame[int, JaxValue]])
        assert_type(
            sum_game_valued.eval(table.players.empty_coalition()),
            FinitePlayerGame[int, JaxValue],
        )
        assert_type(
            SumReductionGame(game_valued_game),
            SumReductionGame[str, FinitePlayerGame[int, JaxValue]],
        )
        SumReductionGame[str, DenseCoalitionGame[int]](game_valued_game)  # ty: ignore[no-matching-overload]

    _ = _check_structured

    def _accept_covariant(game: FinitePlayerGame[str, JaxValue]) -> None:
        assert_type(game.sum(), SumReductionGame[str, JaxValue])

    _accept_covariant(table)

    direct_sum = SumReductionGame(table)
    assert_type(direct_sum, SumReductionGame[str, jax.Array])
    assert_type(direct_sum.eval(table.players.empty_coalition()), jax.Array)

    explicit_sum = SumReductionGame[str, jax.Array](table)
    assert_type(explicit_sum, SumReductionGame[str, jax.Array])
    widened_sum = SumReductionGame[str, JaxValue](table)
    assert_type(widened_sum, SumReductionGame[str, JaxValue])

    # Custom ReductionGame subclass preserves concrete GameT and V.
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class CustomReduction[
        P,
        V: JaxValue = JaxValue,
        GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[Any, Any],
    ](ReductionGame[P, V, GameT, "CustomReduction[P, V, GameT]"]):
        @override
        def _reduce(self, values: Any) -> Any:
            return values

    custom_red = CustomReduction(moebius)
    assert_type(custom_red.game, SparseMoebiusGame[str, jax.Array])

    # Dataclass subclasses of SumReductionGame preserve synthesized constructors.
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class CustomSum(SumReductionGame[Any, Any]):
        extra: int = 0

    custom_sum = CustomSum(table, extra=1)
    assert_type(custom_sum, CustomSum)
    assert_type(custom_sum.extra, int)

    # Invalid construction must be rejected statically.
    _bad_game_type: type[Game[Any, np.ndarray[Any, Any]]]  # ty: ignore[invalid-type-arguments]
    SumReductionGame(weight)  # ty: ignore[no-matching-overload]
    SumReductionGame[int, jax.Array](table)  # ty: ignore[no-matching-overload]
    SumReductionGame[str, jax.Array](moebius)  # ty: ignore[no-matching-overload]
    SumReductionGame[str, jax.Array, DenseCoalitionGame[str]](table)  # ty: ignore[no-matching-overload]
