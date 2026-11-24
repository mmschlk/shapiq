"""Static contracts for ndlike batch operations; checked by the ty lint hook."""

from typing import Any, assert_type

import jax

from shapiq.game.discretederivative import DiscreteDerivative
from shapiq.game.exact import ExactComputer
from shapiq.game.fourier import FourierGame
from shapiq.game.moebius import AdditiveMoebiusGame, SetMoebiusGame
from shapiq.game.table import TableGame
from shapiq.index import BanzhafIndex, ShapleyValueIndex
from shapiq.players import FinitePlayers
from shapiq.sampling._base import SizeWithReplacement
from shapiq.sampling._state import DenseSample, Sample


def batch_results(
    *,
    players: FinitePlayers[str],
    table: TableGame[str],
    exact: ExactComputer,
    derivative: DiscreteDerivative[str, jax.Array],
    fourier: FourierGame[str, jax.Array],
    additive: AdditiveMoebiusGame[str, jax.Array],
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
    assert_type(additive[0].reshape(-1), AdditiveMoebiusGame[str, jax.Array])
    assert_type(banzhaf[0], BanzhafIndex)
    assert_type(shapley[0].reshape(-1), ShapleyValueIndex)
    assert_type(sampler[0], SizeWithReplacement)
    assert_type(sample[0].reshape(-1), DenseSample)
    assert_type(sample.at[0].set(sample[0]), DenseSample)


def accepts_subclasses(additive: AdditiveMoebiusGame[str, jax.Array], dense: DenseSample) -> None:
    game: SetMoebiusGame[str, jax.Array, Any] = additive
    sample: Sample[Any] = dense
    assert_type(game.players, FinitePlayers[str])
    assert_type(sample.coalitions, jax.Array)
