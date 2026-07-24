"""Exact game-space operations: change of basis, projection, fidelity.

Everything here enumerates the full ``2**n`` coalition table and solves
in host float64 — which game a coefficient vector describes is semantic
exactness (the same law that keeps tree routing in float64), while
evaluation precision remains the game boundary's business. The
projection tower — ``project(k) o project(l) == project(k)`` for
``k <= l`` — holds exactly, but only under a shared measure: the measure
is part of the index. ``to_basis`` is the exact change of representation
(all orders, no measure involved); truncated fits under a measure are
``project``, truncated fits from evidence are ``fit_game``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from shapiq.coalitions import DenseCoalitionArray
from shapiq.games._basis import Basis, BasisGame, MoebiusBasis, interaction_terms

if TYPE_CHECKING:
    from shapiq.games._base import Game
    from shapiq.games._measures import Measure

_EXACT_LIMIT = 20


def all_coalitions(n_players: int) -> np.ndarray:
    """Return the full coalition table as host bool masks ``(2**n, n)``.

    Raises:
        ValueError: If the table would exceed ``2**20`` rows; exact
            game-space operations enumerate every coalition and are meant
            for small player counts - sample and estimate beyond that.
    """
    if n_players > _EXACT_LIMIT:
        msg = (
            f"exact game-space operations enumerate 2**{n_players} coalitions; "
            f"they are limited to {_EXACT_LIMIT} players - sample and estimate instead"
        )
        raise ValueError(msg)
    grid = np.arange(2**n_players, dtype=np.int64)[:, None] >> np.arange(n_players)[None, :]
    return (grid & 1).astype(bool)


def _scalar_full_values(game: Game[object]) -> np.ndarray:
    """Evaluate the game on the full table, as a host float64 vector."""
    if game.target_shape != () or game.value_shape != ():
        msg = (
            "exact game-space operations are defined per scalar explanation "
            "target; select a target and value component first"
        )
        raise ValueError(msg)
    masks = all_coalitions(game.n_players)
    host = getattr(game, "_host_values", None)
    if host is not None:  # readable games evaluate exactly off-stack
        return np.asarray(host(masks), dtype=np.float64)
    values = game(DenseCoalitionArray(jnp.asarray(masks)))
    return np.asarray(values, dtype=np.float64)


def to_basis(game: Game[object], basis: Basis) -> BasisGame:
    """Return the game's exact representation in a basis (all orders)."""
    n_players = game.n_players
    truth = _scalar_full_values(game)
    terms = interaction_terms(n_players, n_players)
    atoms = np.asarray(basis.atoms(all_coalitions(n_players), terms, xp=np))
    coefficients, *_ = np.linalg.lstsq(atoms, truth, rcond=None)
    return BasisGame(basis, dict(zip(terms, coefficients, strict=True)), n_players)


def project(game: Game[object], order: int, measure: Measure) -> BasisGame:
    """Return the best game of degree <= ``order`` under the measure.

    The degree-<=k subspace is basis-free; coefficients are reported in
    the moebius basis. Projections onto nested orders compose exactly
    when they share the measure.
    """
    n_players = game.n_players
    truth = _scalar_full_values(game)
    masks = all_coalitions(n_players)
    terms = interaction_terms(n_players, order)
    basis = MoebiusBasis()
    atoms = np.asarray(basis.atoms(masks, terms, xp=np))
    weights = np.sqrt(measure.row_weights(masks))
    coefficients, *_ = np.linalg.lstsq(atoms * weights[:, None], truth * weights, rcond=None)
    return BasisGame(basis, dict(zip(terms, coefficients, strict=True)), n_players)


def fidelity(game: Game[object], surrogate: Game[object], measure: Measure) -> float:
    """Return the surrogate's weighted R^2 against the game under the measure."""
    truth = _scalar_full_values(game)
    guess = _scalar_full_values(surrogate)
    weights = measure.row_weights(all_coalitions(game.n_players))
    mean = float((weights * truth).sum())
    residual = float((weights * (truth - guess) ** 2).sum())
    total = float((weights * (truth - mean) ** 2).sum())
    return 1.0 - residual / total


def shapley_values(game: BasisGame) -> np.ndarray:
    """Return the exact Shapley values of a scalar moebius-basis game.

    ``SV_i = sum over T containing i of m(T) / |T|`` — the coefficient
    read-out that makes explanation addition sound (Shapley values are
    linear in the game).
    """
    coefficients = _scalar_moebius_coefficients(game, "shapley_values")
    values = np.zeros(game.n_players)
    for term, coefficient in zip(game.terms, coefficients, strict=True):
        for player in term:
            values[player] += coefficient / len(term)
    return values


def banzhaf_values(game: BasisGame) -> np.ndarray:
    """Return the exact Banzhaf values of a scalar moebius-basis game.

    ``BV_i = sum over T containing i of m(T) / 2**(|T| - 1)`` — equal to
    twice the degree-one Fourier coefficients.
    """
    coefficients = _scalar_moebius_coefficients(game, "banzhaf_values")
    values = np.zeros(game.n_players)
    for term, coefficient in zip(game.terms, coefficients, strict=True):
        for player in term:
            values[player] += coefficient / 2.0 ** (len(term) - 1)
    return values


def _scalar_moebius_coefficients(game: BasisGame, read_out: str) -> np.ndarray:
    if not isinstance(game.basis, MoebiusBasis):
        msg = (
            f"{read_out} reads moebius coefficients but the game speaks "
            f"{game.basis!r}; convert with to_basis(game, MoebiusBasis()) first"
        )
        raise TypeError(msg)
    coefficients = game.coefficients
    if coefficients.ndim != 1:
        msg = f"{read_out} is a scalar read-out; select a target and value component first"
        raise ValueError(msg)
    return coefficients
