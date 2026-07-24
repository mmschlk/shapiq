"""Parametric games: the intensional tier.

A parametric game is a value function with a readable finite
parametrization in a declared basis. Its coefficients ARE the
attribution/interaction values; evaluating it plays the surrogate. Two
planes live on one object: ``game[T]`` reads a coefficient,
``game(coalitions)`` evaluates the game.

The shipped linear bases are the three logic atoms:

- ``"moebius"`` — AND: atoms fire when all of ``T`` is present; the
  coefficients are Harsanyi dividends (synergy).
- ``"comoebius"`` — OR: atoms fire when any of ``T`` is present; the
  coefficients are the dual game's dividends (redundancy).
- ``"fourier"`` — XOR: parity atoms, orthonormal under the uniform
  measure; Banzhaf values are (scaled) degree-one coefficients.

Sparsity is basis-relative — an OR game is one coefficient in the
comoebius basis and an alternating inclusion-exclusion smear in the
moebius basis — so the basis choice is a modeling statement, and every
parametric game names its own.

Coefficients are stored as host float64: which game a coefficient vector
describes is semantic exactness (like tree routing), while evaluation
follows the stack's precision at the game boundary.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from shapiq.games._base import Game

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping
    from types import ModuleType

    from jax import Array

    from shapiq.coalitions import CoalitionArray

BASES = ("moebius", "comoebius", "fourier")


def interaction_terms(n_players: int, order: int) -> tuple[frozenset[int], ...]:
    """Return every interaction up to ``order`` in canonical order."""
    return tuple(
        frozenset(members)
        for size in range(order + 1)
        for members in combinations(range(n_players), size)
    )


def atom_columns(
    basis: str,
    masks: Array | np.ndarray,
    terms: Iterable[frozenset[int]],
    *,
    xp: ModuleType = jnp,
) -> Array | np.ndarray:
    """Evaluate basis atoms on dense masks ``(..., m, n)`` -> ``(..., m, d)``.

    ``xp`` selects the array backend: the default evaluates on the stack
    (jax, stack precision); exact game-space operations pass numpy so the
    atoms carry host float64 (exactness is semantic there).
    """
    if basis not in BASES:
        msg = f"unknown basis {basis!r}: the shipped bases are {', '.join(BASES)}"
        raise ValueError(msg)
    columns = []
    ones = xp.ones(masks.shape[:-1])
    for term in terms:
        if not term:
            columns.append(ones)
            continue
        inside = masks[..., sorted(term)]
        if basis == "moebius":
            columns.append(inside.all(axis=-1).astype(ones.dtype))
        elif basis == "comoebius":
            columns.append(inside.any(axis=-1).astype(ones.dtype))
        else:  # fourier: chi_T(S) = (-1) ** |T minus S|
            parity = (len(term) - inside.sum(axis=-1)) % 2
            columns.append(1.0 - 2.0 * parity.astype(ones.dtype))
    return xp.stack(columns, axis=-1)


class ParametricGame(Game["Array"]):
    """A game defined by coefficients on basis atoms (intensional tier)."""

    basis: str
    terms: tuple[frozenset[int], ...]

    def __init__(
        self,
        basis: str,
        coefficients: Mapping[Collection[int], float],
        n_players: int,
    ) -> None:
        """Initialize a parametric game from per-interaction coefficients.

        Args:
            basis: One of ``"moebius"``, ``"comoebius"``, ``"fourier"``.
            coefficients: Mapping from interactions (collections of player
                indices) to coefficients. Unlisted interactions are zero.
            n_players: Number of players the game is defined over.

        Raises:
            ValueError: If the basis is unknown, the game has fewer than
                one player, or an interaction names a player out of range.
        """
        if basis not in BASES:
            msg = f"unknown basis {basis!r}: the shipped bases are {', '.join(BASES)}"
            raise ValueError(msg)
        if n_players < 1:
            msg = "a parametric game needs at least one player"
            raise ValueError(msg)
        terms: list[frozenset[int]] = []
        values: list[float] = []
        for interaction, coefficient in coefficients.items():
            term = frozenset(int(player) for player in interaction)
            if term and (min(term) < 0 or max(term) >= n_players):
                msg = f"interaction {tuple(sorted(term))} names a player outside 0..{n_players - 1}"
                raise ValueError(msg)
            terms.append(term)
            values.append(float(coefficient))
        self.basis = basis
        self.terms = tuple(terms)
        self.n_players = n_players
        self.target_shape = ()
        self._coefficients = np.asarray(values, dtype=np.float64)
        self._lookup = {term: index for index, term in enumerate(self.terms)}

    @property
    def coefficients(self) -> np.ndarray:
        """Return the coefficient vector aligned with ``terms`` (read-only)."""
        view = self._coefficients.view()
        view.flags.writeable = False
        return view

    @property
    def support(self) -> tuple[frozenset[int], ...]:
        """Return the interactions carrying nonzero coefficients."""
        return tuple(
            term
            for term, value in zip(self.terms, self._coefficients, strict=True)
            if abs(value) > 1e-12
        )

    @property
    def order(self) -> int:
        """Return the largest interaction size in the support."""
        return max((len(term) for term in self.support), default=0)

    def __getitem__(self, interaction: Collection[int]) -> float:
        """Read one coefficient (the coefficient plane); absent terms are zero."""
        index = self._lookup.get(frozenset(int(player) for player in interaction))
        return 0.0 if index is None else float(self._coefficients[index])

    def _call(self, coalitions: CoalitionArray) -> Array:
        """Evaluate the surrogate (the game plane)."""
        masks = jnp.asarray(coalitions.to_dense())
        atoms = jnp.asarray(atom_columns(self.basis, masks, self.terms))
        return atoms @ jnp.asarray(self._coefficients, dtype=atoms.dtype)

    def _host_values(self, masks: np.ndarray) -> np.ndarray:
        """Evaluate at host float64 for exact game-space operations."""
        atoms = np.asarray(atom_columns(self.basis, masks, self.terms, xp=np))
        return atoms @ self._coefficients

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(basis={self.basis!r}, n_players={self.n_players!r}, "
            f"order={self.order!r}, n_terms={len(self.terms)!r})"
        )
