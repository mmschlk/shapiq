"""Fit an intensional surrogate from evidence: the game-fitter verb.

The fit-then-read estimator family learns a parametric game from
evaluated coalitions and reads it exactly (proxy models, sparse
recovery, trees). This is its simplest member: an unweighted
least-squares fit onto a basis truncation, at host float64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.games._basis import Basis, BasisGame, MoebiusBasis, interaction_terms

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def fit_game(
    masks: ArrayLike,
    values: ArrayLike,
    n_players: int,
    *,
    order: int,
    basis: Basis | None = None,
) -> BasisGame:
    """Fit a degree-<= ``order`` parametric game to evaluated coalitions.

    Args:
        masks: Dense coalition masks of shape ``(m, n_players)``.
        values: The evaluated game values aligned with the masks.
        n_players: Number of players of the fitted game.
        order: Largest interaction size in the fit.
        basis: The basis to fit in (moebius by default).

    Returns:
        The least-squares parametric game; its coefficients are exact
        when the evidence identifies the truncation.
    """
    basis = MoebiusBasis() if basis is None else basis
    host_masks = np.asarray(masks, dtype=bool).reshape(-1, n_players)
    host_values = np.asarray(values, dtype=np.float64).reshape(-1)
    terms = interaction_terms(n_players, order)
    atoms = np.asarray(basis.atoms(host_masks, terms, xp=np))
    coefficients, *_ = np.linalg.lstsq(atoms, host_values, rcond=None)
    return BasisGame(basis, dict(zip(terms, coefficients, strict=True)), n_players)
