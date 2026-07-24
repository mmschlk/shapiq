"""Bases and basis games: the readable tier of the value-function world.

A basis is a value object that owns its atoms — the same move that made
indices instances (parameters live on values, extension means writing a
new value, never registering a string). A ``BasisGame`` is a game known
through a finite coefficient vector on a declared basis: evaluating it
plays the surrogate (the game plane), indexing it reads a coefficient
(the coefficient plane). Every game the library produces as output —
explanations included — is one of these, possibly with provenance.

The shipped bases are the three logic atoms: AND (``MoebiusBasis`` —
synergy dividends), OR (``CoMoebiusBasis`` — redundancy, the dual game's
dividends), XOR (``FourierBasis`` — parity, orthonormal under the
uniform measure; Banzhaf values are its scaled degree-one coefficients).
Sparsity is basis-relative, so declaring the basis is a modeling
statement.

Coefficients are stored as host float64 in the canonical leading layout
(value axes, then target axes, term axis last): which game a coefficient
vector describes is semantic exactness; evaluation follows the stack's
precision at the game boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np

from shapiq.games._base import Game

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping
    from types import ModuleType

    from jax import Array

    from shapiq.coalitions import CoalitionArray


def interaction_terms(n_players: int, order: int) -> tuple[frozenset[int], ...]:
    """Return every interaction up to ``order`` in canonical order."""
    return tuple(
        frozenset(members)
        for size in range(order + 1)
        for members in combinations(range(n_players), size)
    )


@runtime_checkable
class Basis(Protocol):
    """A basis of the game space: a value that evaluates its own atoms."""

    def atoms(
        self,
        masks: Array | np.ndarray,
        terms: Iterable[frozenset[int]],
        *,
        xp: ModuleType = ...,
    ) -> Array | np.ndarray:
        """Evaluate one atom column per term on dense masks ``(..., m, n)``."""
        ...


def _term_columns(
    masks: Array | np.ndarray,
    terms: Iterable[frozenset[int]],
    xp: ModuleType,
    column: Callable[[Array | np.ndarray, int, np.dtype], Array | np.ndarray],
) -> Array | np.ndarray:
    ones = xp.ones(masks.shape[:-1])
    columns = []
    for term in terms:
        if not term:
            columns.append(ones)
            continue
        columns.append(column(masks[..., sorted(term)], len(term), ones.dtype))
    return xp.stack(columns, axis=-1)


@dataclass(frozen=True)
class MoebiusBasis:
    """AND atoms: a term fires when all of its players are present."""

    def atoms(
        self,
        masks: Array | np.ndarray,
        terms: Iterable[frozenset[int]],
        *,
        xp: ModuleType = jnp,
    ) -> Array | np.ndarray:
        """Evaluate unanimity atoms."""
        return _term_columns(
            masks, terms, xp, lambda inside, _size, dtype: inside.all(axis=-1).astype(dtype)
        )


@dataclass(frozen=True)
class CoMoebiusBasis:
    """OR atoms: a term fires when any of its players is present."""

    def atoms(
        self,
        masks: Array | np.ndarray,
        terms: Iterable[frozenset[int]],
        *,
        xp: ModuleType = jnp,
    ) -> Array | np.ndarray:
        """Evaluate redundancy atoms."""
        return _term_columns(
            masks, terms, xp, lambda inside, _size, dtype: inside.any(axis=-1).astype(dtype)
        )


@dataclass(frozen=True)
class FourierBasis:
    """XOR atoms: parity characters, orthonormal under the uniform measure."""

    def atoms(
        self,
        masks: Array | np.ndarray,
        terms: Iterable[frozenset[int]],
        *,
        xp: ModuleType = jnp,
    ) -> Array | np.ndarray:
        """Evaluate parity atoms ``chi_T(S) = (-1) ** |T minus S|``."""

        def column(
            inside: Array | np.ndarray, size: int, dtype: np.dtype
        ) -> Array | np.ndarray:
            parity = (size - inside.sum(axis=-1)) % 2
            return 1.0 - 2.0 * parity.astype(dtype)

        return _term_columns(masks, terms, xp, column)


class BasisGame(Game["Array"]):
    """A game known through coefficients on a declared basis.

    Two planes on one object: ``game(coalitions)`` evaluates the
    surrogate, ``game[T]`` reads a coefficient. Unlisted interactions are
    zero — the empty interaction included: it is an ordinary slot, and
    what it holds (a baseline, a fitted intercept, nothing) is decided by
    whoever builds the game.
    """

    basis: Basis
    terms: tuple[frozenset[int], ...]

    def __init__(
        self,
        basis: Basis,
        coefficients: Mapping[Collection[int], object] | None,
        n_players: int,
        *,
        terms: tuple[frozenset[int], ...] | None = None,
        values: np.ndarray | None = None,
        value_shape: tuple[int, ...] = (),
        target_shape: tuple[int, ...] = (),
    ) -> None:
        """Initialize from a coefficient mapping or aligned terms/values.

        Args:
            basis: The basis the coefficients live in.
            coefficients: Mapping from interactions to coefficients
                (scalars, or arrays of ``(*value_shape, *target_shape)``).
                Duplicate spellings of one interaction are summed. Pass
                ``None`` when providing ``terms``/``values`` directly.
            n_players: Number of players of the game.
            terms: Interactions aligned with ``values`` (builder path).
            values: Host float64 coefficients in the canonical leading
                layout ``(*value_shape, *target_shape, n_terms)``.
            value_shape: The game's value shape.
            target_shape: The game's explanation-target shape.
        """
        if n_players < 1:
            msg = "a basis game needs at least one player"
            raise ValueError(msg)
        if coefficients is not None:
            merged: dict[frozenset[int], np.ndarray] = {}
            for interaction, coefficient in coefficients.items():
                term = frozenset(int(player) for player in interaction)
                if term and (min(term) < 0 or max(term) >= n_players):
                    msg = (
                        f"interaction {tuple(sorted(term))} names a player "
                        f"outside 0..{n_players - 1}"
                    )
                    raise ValueError(msg)
                block = np.asarray(coefficient, dtype=np.float64)
                merged[term] = merged.get(term, 0.0) + block
            terms = tuple(merged)
            stacked = (
                np.stack([merged[term] for term in terms], axis=-1)
                if terms
                else np.zeros((*value_shape, *target_shape, 0))
            )
            values = stacked
        if terms is None or values is None:
            msg = "provide either a coefficient mapping or terms and values"
            raise ValueError(msg)
        self.basis = basis
        self.terms = terms
        self.n_players = n_players
        self.value_shape = value_shape
        self.target_shape = target_shape
        self._values = np.asarray(values, dtype=np.float64)
        self._lookup = {term: position for position, term in enumerate(terms)}

    @property
    def coefficients(self) -> np.ndarray:
        """Return the coefficient array ``(*value, *target, n_terms)``, read-only."""
        view = self._values.view()
        view.flags.writeable = False
        return view

    @property
    def support(self) -> tuple[frozenset[int], ...]:
        """Return the interactions carrying any nonzero coefficient."""
        flat = self._values.reshape(-1, self._values.shape[-1])
        alive = np.abs(flat).max(axis=0) > 1e-12 if flat.size else np.zeros(0, dtype=bool)
        return tuple(term for term, keep in zip(self.terms, alive, strict=True) if keep)

    @property
    def order(self) -> int:
        """Return the largest interaction size in the support."""
        return max((len(term) for term in self.support), default=0)

    def interactions(self, order: int | None = None) -> tuple[frozenset[int], ...]:
        """Return stored interactions, optionally only those of one size."""
        if order is None:
            return self.terms
        return tuple(term for term in self.terms if len(term) == order)

    def __getitem__(self, interaction: Collection[int]) -> np.ndarray | float:
        """Read one coefficient (the coefficient plane); absent terms are zero.

        Returns a float for scalar games, else an array of
        ``(*target_shape, *value_shape)`` (the public trailing layout).
        """
        if isinstance(interaction, int):
            msg = (
                f"interactions are player collections: read player "
                f"{interaction} with game[({interaction},)]"
            )
            raise TypeError(msg)
        position = self._lookup.get(frozenset(int(player) for player in interaction))
        if position is None:
            block = np.zeros((*self.value_shape, *self.target_shape))
        else:
            block = self._values[..., position]
        if block.ndim == 0:
            return float(block)
        n_value = len(self.value_shape)
        if n_value == 0:
            return block
        # public reads trail: targets first, then value axes
        return np.moveaxis(block, range(n_value), range(-n_value, 0))

    def _call(self, coalitions: CoalitionArray) -> Array:
        """Evaluate the surrogate (the game plane)."""
        masks = jnp.asarray(coalitions.to_dense())
        n_samples = masks.shape[-2]
        if not self.terms:
            leading = jnp.broadcast_shapes(self.target_shape, masks.shape[:-2])
            return jnp.zeros((*leading, n_samples, *self.value_shape))
        n_terms = len(self.terms)
        atoms = jnp.asarray(self.basis.atoms(masks, self.terms))  # (..., m, d)
        coefficients = jnp.asarray(self._values, dtype=atoms.dtype)
        if not self.target_shape:
            flat = coefficients.reshape(-1, n_terms)  # (Vn, d); Vn = 1 when scalar
            combined = atoms @ flat.T  # (..., m, Vn)
            if not self.value_shape:
                return combined[..., 0]
            return combined.reshape(*combined.shape[:-1], *self.value_shape)
        # target axes: each target evaluates its own coefficient slice
        n_flat_targets = int(np.prod(self.target_shape))
        n_flat_values = int(np.prod(self.value_shape)) if self.value_shape else 1
        broadcast = jnp.broadcast_to(
            atoms, (*self.target_shape, n_samples, n_terms)
        ).reshape(n_flat_targets, n_samples, n_terms)
        per_target = coefficients.reshape(n_flat_values, n_flat_targets, n_terms)
        combined = jnp.einsum("tmd,vtd->tmv", broadcast, per_target)
        if not self.value_shape:
            return combined[..., 0].reshape(*self.target_shape, n_samples)
        return combined.reshape(*self.target_shape, n_samples, *self.value_shape)

    def _host_values(self, masks: np.ndarray) -> np.ndarray:
        """Evaluate at host float64 for exact game-space operations."""
        if not self.terms:
            return np.zeros(masks.shape[:-1])
        atoms = np.asarray(self.basis.atoms(masks, self.terms, xp=np))
        if self._values.ndim != 1:
            msg = "exact game-space operations are scalar; select a component first"
            raise ValueError(msg)
        return atoms @ self._values

    def detach(self) -> BasisGame:
        """Return the bare basis game, shedding any provenance a subclass adds."""
        return BasisGame(
            self.basis,
            None,
            self.n_players,
            terms=self.terms,
            values=self._values,
            value_shape=self.value_shape,
            target_shape=self.target_shape,
        )

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(basis={self.basis!r}, n_players={self.n_players!r}, "
            f"order={self.order!r}, n_terms={len(self.terms)!r})"
        )
