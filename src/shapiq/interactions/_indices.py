"""Interaction index singletons, their capability protocols, and generalizations."""

from __future__ import annotations

from functools import cache
from math import comb
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Literal,
    NoReturn,
    Protocol,
    Self,
    cast,
    get_args,
    runtime_checkable,
)

import jax.numpy as jnp

from shapiq._shape import validate_int

if TYPE_CHECKING:
    from jax import Array

    from shapiq.interactions._types import OrderSemantics


@runtime_checkable
class InteractionIndex(Protocol):
    """An interaction index, represented as a module-level singleton value.

    Indices are passed to explainers as the value itself — ``SII``, never
    ``SII(...)`` — and carry no order: the explanation order is given to the
    explainer, which resolves it through ``resolve_order``. Explainers select
    behavior by index identity and capability, never by name; the ``name`` is
    explanation metadata. The order semantics record whether the order given
    to the explainer is explanation coverage (attributions of shared
    interactions are unchanged across orders, as for SII and BII) or part of
    the index identity (attributions change with the order, as for STII and
    FSII). Indices that generalize a probabilistic value declare it: their
    order-1 explanations equal that value's, and the declaration is tested
    numerically. Third-party indices need not inherit ``Index``: any
    conforming value works with the capability-dispatched explainers.
    """

    @property
    def name(self) -> str:
        """Return the name recorded on explanations."""
        ...

    @property
    def order_semantics(self) -> OrderSemantics:
        """Return whether the explanation order is coverage or index identity."""
        ...

    @property
    def includes_empty_interaction(self) -> bool:
        """Return whether explanations carry an order-0 attribution."""
        ...

    @property
    def preserves_value(self) -> bool:
        """Return whether order-1 attributions are identical at every explanation order.

        For a generalizing index this means the generalized value stays
        readable off the order-1 attributions at any order; indices with no
        generalized value preserve trivially through coverage semantics.
        """
        ...

    @property
    def generalizes(self) -> Index[Literal["SV", "BV"]] | None:
        """Return the probabilistic value whose explanations equal this index's at order 1."""
        ...

    def resolve_order(self, order: int | None, *, n_players: int) -> int:
        """Validate and resolve the explanation order for a game."""
        ...


@runtime_checkable
class CardinalInteractionIndex(InteractionIndex, Protocol):
    """Capability: attributions are cardinality-weighted discrete derivatives.

    Cardinal interaction indices assign to an interaction ``S`` a weighted
    sum of its discrete derivatives over outside coalitions ``T``, with
    weights depending only on the cardinalities of ``S`` and ``T``.
    Cardinality-only dependence — anonymity — is the symmetry shared by every
    shipped index, though the regression and aggregation families realize it
    through symmetric kernels and bases rather than this weighted-derivative
    form; player-specific weightings (weighted Shapley values, Owen values
    under a coalition structure) are a separate future capability, not a
    generalization of this one.
    """

    @property
    def min_interaction_size(self) -> int:
        """Return the smallest represented interaction size.

        Zero when the rule attributes to the empty interaction on the
        centered game, as for the Co-Moebius transform.
        """
        ...

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return one weight per outside-coalition size ``0..n - s``.

        Coverage-semantics indices return identical weights at every order;
        identity-semantics indices (such as STII) depend on it.
        """
        ...


@runtime_checkable
class GeneralizedValueIndex(InteractionIndex, Protocol):
    """Capability: attributions are cardinality-weighted bloc marginals.

    Generalized values assign to an interaction ``S`` a weighted sum of the
    marginal contributions ``v(T | S) - v(T)`` of the whole interaction
    joining outside coalitions ``T``, with weights depending only on
    cardinalities.
    """

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return one weight per outside-coalition size ``0..n - s``."""
        ...


@runtime_checkable
class RegressionIndex(InteractionIndex, Protocol):
    """Capability: attributions solve a kernel-weighted least squares fit."""

    def regression_kernel(self, n_players: int) -> Array:
        """Return one kernel weight per coalition size ``0..n``.

        A zero weight at the empty or grand coalition marks that coalition
        as an exact interpolation constraint rather than a weighted row;
        nonzero end weights (the uniform Banzhaf kernel) mean every
        coalition is fitted and the fit carries a free intercept.
        """
        ...


class Index[NameT: str]:
    """An interaction index as a module-level singleton value.

    Every shipped index is the only instance of a hidden class parameterized
    by its name literal — ``Index[Literal["SII"]]`` — so the literal in the
    type is the single source of the runtime name, closed sets of indices are
    ordinary type expressions such as ``Index[Literal["SV", "FSII", "FBII"]]``,
    and the exported value (``SII``) is at once what users pass, what dispatch
    compares against with ``is``, and what explanations record. Constructing
    the hidden class returns the singleton, so ``copy``, ``deepcopy``, and
    pickling all preserve identity.
    """

    _name: ClassVar[str]
    _instance: ClassVar[Index[str] | None] = None

    order_semantics: ClassVar[OrderSemantics]
    preserves_value: ClassVar[bool]
    includes_empty_interaction: ClassVar[bool]
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Derive the index name from the ``Index[Literal[...]]`` base."""
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            for argument in get_args(base):
                literals = get_args(argument)
                if literals and all(isinstance(entry, str) for entry in literals):
                    cls._name = literals[0]
                    return

    def __new__(cls) -> Self:
        """Return the class's only instance, creating it on first use."""
        existing = cls.__dict__.get("_instance")
        if existing is not None:
            return cast("Self", existing)
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __call__(self, *args: object, **kwargs: object) -> NoReturn:
        """Reject calls with a teaching error: the value is already the index."""
        del args, kwargs
        symbol = type(self).__name__.removeprefix("_")
        msg = (
            f"{symbol} is already the index — interaction indices are singleton "
            f"values, never constructed: pass it to an explainer, e.g. "
            f"ExactExplainer(game, {symbol}{self._order_example()})"
        )
        raise TypeError(msg)

    def __repr__(self) -> str:
        """Print the singleton's exported name."""
        return type(self).__name__.removeprefix("_")

    @property
    def name(self) -> NameT:
        """Return the name recorded on explanations, from the type's literal."""
        return cast("NameT", self._name)

    def resolve_order(self, order: int | None, *, n_players: int) -> int:
        """Validate and resolve the explanation order for a game."""
        raise NotImplementedError

    def _order_example(self) -> str:
        """Return ``", order=2"`` when the index needs an explicit order."""
        try:
            self.resolve_order(None, n_players=2)
        except (TypeError, NotImplementedError):
            return ", order=2"
        return ""


class _OrderOneIndex[NameT: str](Index[NameT]):
    """Order resolution for probabilistic values: the order is fixed at one."""

    def resolve_order(self, order: int | None, *, n_players: int) -> int:
        """Return ``1``, rejecting any other explicitly requested order."""
        if n_players < 1:
            msg = f"order must not exceed the number of players, got 1 for {n_players}"
            raise ValueError(msg)
        if order is None:
            return 1
        validate_int("order", order, minimum=1)
        if order != 1:
            msg = (
                f"{self.name} attributes to single players only: "
                f"order is fixed at 1, got {order}"
            )
            raise ValueError(msg)
        return 1


class _ExplicitOrderIndex[NameT: str](Index[NameT]):
    """Order resolution for interaction indices: an explicit order is required."""

    def resolve_order(self, order: int | None, *, n_players: int) -> int:
        """Validate an explicitly given order against the player count."""
        if order is None:
            msg = (
                f"{self.name} explanations need an explicit order: "
                f"pass order= to the explainer, e.g. order=2"
            )
            raise TypeError(msg)
        validate_int("order", order, minimum=1)
        if order > n_players:
            msg = f"order must not exceed the number of players, got {order} for {n_players}"
            raise ValueError(msg)
        return order


class _AllOrdersIndex[NameT: str](Index[NameT]):
    """Order resolution for transforms: the order defaults to all players."""

    def resolve_order(self, order: int | None, *, n_players: int) -> int:
        """Return the player count when no order is given."""
        if order is None:
            return n_players
        validate_int("order", order, minimum=1)
        if order > n_players:
            msg = f"order must not exceed the number of players, got {order} for {n_players}"
            raise ValueError(msg)
        return order


class _SV(_OrderOneIndex[Literal["SV"]]):
    """The Shapley value: the unique efficient attribution to single players."""

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = None

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _shapley_derivative_weights(n_players, interaction_size)

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Shapley bloc-marginal weights for single players.

        The Shapley value is the singleton restriction of the Shapley
        generalized value: for an interaction of size one the discrete
        derivative and the bloc marginal coincide, so SV satisfies the
        generalized-value capability there and nowhere else.
        """
        del order
        if interaction_size != 1:
            msg = f"SV attributes to single players only, got interaction_size={interaction_size}"
            raise ValueError(msg)
        return _shapley_derivative_weights(n_players, 1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends.

        The Shapley value is also the constrained kernel regression of order
        one (KernelSHAP), so SV satisfies both capabilities; explainers with
        a cheaper discrete-derivative path prefer it.
        """
        return _shapley_regression_kernel(n_players)


SV: Final = _SV()


class _BV(_OrderOneIndex[Literal["BV"]]):
    """The Banzhaf value: uniform-coalition attribution to single players."""

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = None

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _banzhaf_derivative_weights(n_players, interaction_size)

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Banzhaf bloc-marginal weights for single players.

        The Banzhaf value is the singleton restriction of the Banzhaf
        generalized value, mirroring SV's dual capability.
        """
        del order
        if interaction_size != 1:
            msg = f"BV attributes to single players only, got interaction_size={interaction_size}"
            raise ValueError(msg)
        return _banzhaf_derivative_weights(n_players, 1)


BV: Final = _BV()


class _SII(_ExplicitOrderIndex[Literal["SII"]]):
    """The Shapley interaction index; generalizes SV.

    The explanation order is coverage: attributions of shared interactions
    are identical across orders, so a higher order only adds interactions.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _shapley_derivative_weights(n_players, interaction_size)


SII: Final = _SII()


class _BII(_ExplicitOrderIndex[Literal["BII"]]):
    """The Banzhaf interaction index; generalizes BV.

    The explanation order is coverage: attributions of shared interactions
    are identical across orders, so a higher order only adds interactions.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = BV

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _banzhaf_derivative_weights(n_players, interaction_size)


BII: Final = _BII()


class _CHII(_ExplicitOrderIndex[Literal["CHII"]]):
    """The chaining interaction index; generalizes SV.

    The explanation order is coverage: attributions of shared interactions
    are identical across orders. The chaining weights treat the interaction
    as the head of a chain of arrivals.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return chaining discrete-derivative weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _chaining_derivative_weights(n_players, interaction_size)


CHII: Final = _CHII()


class _STII(_ExplicitOrderIndex[Literal["STII"]]):
    """The Shapley-Taylor interaction index; generalizes SV.

    The explanation order is part of the index identity: attributions of
    shared interactions change with the order. Interactions below the top
    order are discrete derivatives at the empty coalition; the top order
    distributes the remaining game mass along permutations.
    """

    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return anchored weights below the top order, Taylor weights at it."""
        if interaction_size < order:
            return _empty_anchor_weights(n_players, interaction_size)
        return _taylor_top_weights(n_players, interaction_size)


STII: Final = _STII()


class _KSII(_ExplicitOrderIndex[Literal["k-SII"]]):
    """The efficient k-Shapley interaction index; generalizes SV.

    The explanation order is part of the index identity: k-SII aggregates
    Shapley interactions of all orders up to the explanation order into an
    efficient explanation via Bernoulli numbers, so attributions of shared
    interactions change with the order.
    """

    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV


KSII: Final = _KSII()


class _FSII(_ExplicitOrderIndex[Literal["FSII"]]):
    """The faithful Shapley interaction index; generalizes SV.

    The explanation order is part of the index identity: the index is the
    best order-additive approximation of the game under the Shapley kernel,
    interpolating the empty and grand coalition exactly.
    """

    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends."""
        return _shapley_regression_kernel(n_players)


FSII: Final = _FSII()


class _FBII(_ExplicitOrderIndex[Literal["FBII"]]):
    """The faithful Banzhaf interaction index; generalizes BV.

    The explanation order is part of the index identity: the index is the
    best order-additive approximation of the game under the uniform kernel.
    The fit is unconstrained with a free intercept, so the order-0
    attribution is the fitted intercept of the centered game rather than
    zero; the empty-coalition value itself is the explanation's baseline.
    """

    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = True
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = BV

    def regression_kernel(self, n_players: int) -> Array:
        """Return the uniform Banzhaf kernel: unit weight for every size.

        Nonzero weights at the empty and grand coalition mean those rows are
        fitted like any other rather than interpolated as constraints, and
        the fit carries a free intercept as the order-0 attribution.
        """
        return jnp.ones(n_players + 1)


FBII: Final = _FBII()


class _KADDSHAP(_ExplicitOrderIndex[Literal["kADD-SHAP"]]):
    """The k-additive Shapley index; generalizes SV.

    The explanation order is part of the index identity: the index fits an
    order-additive game in the Bernoulli-weighted interaction basis under
    the Shapley kernel, interpolating the grand coalition exactly, so its
    order-1 attributions remain Shapley values at every order.
    """

    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends."""
        return _shapley_regression_kernel(n_players)


KADDSHAP: Final = _KADDSHAP()


class _Moebius(_AllOrdersIndex[Literal["Moebius"]]):
    """The Moebius transform: the game's interaction masses themselves.

    A cardinal index anchored at the empty coalition; without an explicit
    order the explanation represents every interaction up to the grand
    coalition. On the centered game the empty mass is zero, so the
    empty-coalition value is carried as the explanation's baseline rather
    than as an attribution.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = None

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return discrete-derivative-at-empty weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _empty_anchor_weights(n_players, interaction_size)


Moebius: Final = _Moebius()


class _CoMoebius(_AllOrdersIndex[Literal["Co-Moebius"]]):
    """The Co-Moebius transform: derivatives anchored at the complement.

    A cardinal index whose derivative of an interaction is taken at the
    coalition of all remaining players; without an explicit order the
    explanation represents every interaction. Its order-0 attribution on
    the centered game is the grand total ``v(N) - v(empty)``.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = True
    min_interaction_size: ClassVar[int] = 0
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = None

    def derivative_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return discrete-derivative-at-complement weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _grand_anchor_weights(n_players, interaction_size)


CoMoebius: Final = _CoMoebius()


class _SGV(_ExplicitOrderIndex[Literal["SGV"]]):
    """The Shapley generalized value; generalizes SV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions with Shapley weights. The explanation order
    is coverage.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Shapley bloc-marginal weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _shapley_derivative_weights(n_players, interaction_size)


SGV: Final = _SGV()


class _BGV(_ExplicitOrderIndex[Literal["BGV"]]):
    """The Banzhaf generalized value; generalizes BV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions uniformly. The explanation order is coverage.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = BV

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return Banzhaf bloc-marginal weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _banzhaf_derivative_weights(n_players, interaction_size)


BGV: Final = _BGV()


class _CHGV(_ExplicitOrderIndex[Literal["CHGV"]]):
    """The chaining generalized value; generalizes SV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions with chaining weights. The explanation order
    is coverage.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return chaining bloc-marginal weights per outside size."""
        del order  # coverage semantics: identical weights at every order
        return _chaining_derivative_weights(n_players, interaction_size)


CHGV: Final = _CHGV()


class _IGV(_ExplicitOrderIndex[Literal["IGV"]]):
    """The internal generalized value.

    The attribution of an interaction is its stand-alone worth over the
    empty coalition, ``v(S) - v(empty)``. The explanation order is coverage.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = None

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return bloc-marginal weights anchored at the empty coalition."""
        del order  # coverage semantics: identical weights at every order
        return _empty_anchor_weights(n_players, interaction_size)


IGV: Final = _IGV()


class _EGV(_ExplicitOrderIndex[Literal["EGV"]]):
    """The external generalized value.

    The attribution of an interaction is its contribution on top of all
    remaining players, ``v(N) - v(N without S)``. The explanation order is
    coverage.
    """

    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = None

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return bloc-marginal weights anchored at the complement."""
        del order  # coverage semantics: identical weights at every order
        return _grand_anchor_weights(n_players, interaction_size)


EGV: Final = _EGV()


class _JointSV(_ExplicitOrderIndex[Literal["JointSV"]]):
    """The joint Shapley value; generalizes SV.

    Attributions weight bloc marginal contributions with arrival-process
    weights in which coalitions of up to the explanation order arrive
    together, so the explanation is efficient. The explanation order is part
    of the index identity.
    """

    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[Index[Literal["SV", "BV"]] | None] = SV

    def marginal_weights(self, n_players: int, interaction_size: int, *, order: int) -> Array:
        """Return joint-arrival bloc-marginal weights per outside size."""
        return _joint_arrival_weights(n_players, order)[: n_players - interaction_size + 1]


JointSV: Final = _JointSV()


def _shapley_regression_kernel(n_players: int) -> Array:
    """Return the Shapley kernel per coalition size with zero-weight endpoints."""
    interior = [
        1.0 / ((n_players - 1) * comb(n_players - 2, size - 1)) for size in range(1, n_players)
    ]
    return jnp.asarray([0.0, *interior, 0.0])


def _shapley_derivative_weights(n_players: int, size: int) -> Array:
    """Return SII discrete-derivative weights per outside-coalition size."""
    free = n_players - size
    return jnp.asarray([1.0 / ((free + 1) * comb(free, t)) for t in range(free + 1)])


def _banzhaf_derivative_weights(n_players: int, size: int) -> Array:
    """Return BII discrete-derivative weights per outside-coalition size."""
    free = n_players - size
    return jnp.full(free + 1, 2.0**-free)


def _chaining_derivative_weights(n_players: int, size: int) -> Array:
    """Return CHII discrete-derivative weights per outside-coalition size."""
    return jnp.asarray(
        [size / ((size + t) * comb(n_players, size + t)) for t in range(n_players - size + 1)],
    )


def _empty_anchor_weights(n_players: int, size: int) -> Array:
    """Return discrete-derivative-at-empty weights per outside-coalition size."""
    return jnp.zeros(n_players - size + 1).at[0].set(1.0)


def _grand_anchor_weights(n_players: int, size: int) -> Array:
    """Return discrete-derivative-at-complement weights per outside-coalition size."""
    return jnp.zeros(n_players - size + 1).at[-1].set(1.0)


def _taylor_top_weights(n_players: int, size: int) -> Array:
    """Return top-order Shapley-Taylor weights per outside-coalition size."""
    return jnp.asarray(
        [size / (n_players * comb(n_players - 1, t)) for t in range(n_players - size + 1)],
    )


@cache
def _joint_arrival_weights(n_players: int, order: int) -> Array:
    """Return joint Shapley weights per coalition size from the arrival recursion."""
    weights = [0.0] * n_players
    weights[0] = 1.0 / sum(comb(n_players, size) for size in range(1, order + 1))
    for reached in range(1, n_players):
        largest_step = min(order, n_players - reached)
        smallest_origin = max(reached - order, 0)
        denominator = sum(comb(n_players - reached, size) for size in range(1, largest_step + 1))
        numerator = sum(
            comb(reached, size) * weights[size] for size in range(smallest_origin, reached)
        )
        weights[reached] = numerator / denominator
    return jnp.asarray(weights)
