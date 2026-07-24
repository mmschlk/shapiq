"""The gradient bridge: extensions of games to the cube.

A game's smooth extension is any differentiable function on ``[0, 1]^n``
agreeing with it at the vertices. Gradient explainers integrate a
gradient along a path of an extension — and which extension is part of
the method: Integrated Gradients on the model's own extension is IG; the
same diagonal integral on the *multilinear* extension is exactly the
Shapley value (Owen's theorem), the center gradient is the Banzhaf
value, and center mixed partials are Banzhaf interactions. Two methods
can agree on the game at every vertex and still attribute differently —
completeness does not pin the answer, the extension does.

Intensional games give the multilinear extension analytically: for a
moebius game ``g(z) = sum_T m(T) prod_{i in T} z_i``, so its diagonal
gradient is closed form and Owen's theorem is a two-line identity.
Everything here is host float64 (exactness is semantic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.games._projection import _scalar_moebius_coefficients

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.games._basis import BasisGame


def multilinear_diagonal_gradient(game: BasisGame, t: float) -> np.ndarray:
    """Gradient of the multilinear extension on the diagonal ``z = t * ones``.

    ``dg/dz_i = sum over T containing i of m(T) * t ** (|T| - 1)`` — the
    intensional tier's free gradient oracle.

    Raises:
        ValueError: If the game is not in the moebius basis; convert with
            ``to_basis(game, MoebiusBasis())`` first.
    """
    coefficients = _scalar_moebius_coefficients(game, "multilinear_diagonal_gradient")
    gradient = np.zeros(game.n_players)
    for term, coefficient in zip(game.terms, coefficients, strict=True):
        for player in term:
            gradient[player] += coefficient * t ** (len(term) - 1)
    return gradient


def integrated_gradients(
    gradient_on_diagonal: Callable[[float], np.ndarray],
    n_players: int,
    steps: int,
) -> np.ndarray:
    """Midpoint-rule diagonal path integral of a gradient oracle.

    The gradient-explainer verb: the budget is gradient evaluations.
    Applied to a game's multilinear extension this returns its Shapley
    values (Owen's theorem); applied to any other extension of the same
    game it returns that extension's integrated gradients.
    """
    midpoints = (np.arange(steps) + 0.5) / steps
    total = np.zeros(n_players)
    for t in midpoints:
        total += gradient_on_diagonal(float(t))
    return total / steps
