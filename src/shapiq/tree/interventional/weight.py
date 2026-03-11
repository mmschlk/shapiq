"""Weight functions for interventional tree explainer interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Callable

from scipy.special import beta, binom, factorial

from shapiq.utils.sets import powerset


def moebius_value(S: set, A: set, B: set) -> float:
    """Compute the Möbius value for given sets S, A, and B.

    Args:
        S: The set S.
        A: The set A.
        B: The set B.

    Returns:
        The Möbius value.
    """
    return sum(
        [
            (-1) ** (len(S) - len(T)) * (1 if A.issubset(set(T)) and set(T).issubset(B) else 0)
            for T in powerset(S)
        ]
    )


def general_weight_fbii(A: set, B: set, N: set, U: set, max_order: int) -> float:
    """Compute the general weight for FBII for given sets A, B, N and U.

    Args:
        A: Set A.
        B: Set B.
        N: Set of all players.
        U: Current coalition.
        max_order: Maximum order of interactions.

    Returns:
        The general weight for FBII.
    """
    # Make sure that A,B,N,U contain integers
    A = set(map(int, A))
    B = set(map(int, B))
    N = set(map(int, N))
    U = set(map(int, U))

    u_0 = len(U.intersection(N.difference(B)))
    a = len(A)
    b = len(B)
    n = len(N)
    u = len(U)

    w1 = (-1) ** (u_0) if A.issubset(U) else 0

    w = 0
    for level in range(max_order + 1 - u_0 - a, n - b - u_0 + 1):
        term = (
            (-1) ** (u_0 + level + max_order - u)
            * (1 / 2) ** (a + level + u_0 - u)
            * binom(a + level + u_0 - u - 1, max_order - u)
            * binom(n - b - u_0, level)
        )
        w += term
    return w1 + w


def shapley_weight_function(a: int, b: int) -> float:
    """Compute the Shapley weight for given set sizes a and b.

    Args:
        a: Size of set A.
        b: Size of set B.

    Returns:
        The Shapley weight.
    """
    return 1.0 / ((a + b + 1) * binom(a + b, b))


def shapley_based_weight_function(A: set, B: set, N: set, U: set) -> float:
    """Compute the Shapley based weight for given sets A, B, N and U.

    Args:
        A: Set A.
        B: Set B.
        N: Set of all players.
        U: Current coalition.

    Returns:
        The Shapley based weight.
    """
    a = len(A) - len(B.intersection(U))
    b = len(N.difference(B.union(U)))
    if a < 0 or b < 0:
        warn(f"Invalid sets A={A}, B={B}, N={N}, U={U}. Returning 0 instead.", stacklevel=2)
        return 0
    sign = (-1) ** (len(U.intersection(N.difference(B))))
    return sign * 1.0 / ((a + b + 1) * binom(a + b, b))


def banzhaf_weight_function(A: set, B: set, N: set, U: set) -> float:
    """Compute the Banzhaf based weight for given sets A, B, N and U.

    Args:
        A: Set A.
        B: Set B.
        N: Set of all players.
        U: Current coalition.

    Returns:
        The Banzhaf based weight.
    """
    sign = (-1) ** (len(U.intersection(N.difference(B))))
    weight = 1.0 / (2 ** (len(N) + len(A) - len(B) - len(U)))
    return sign * weight


def chaining_weight_function(A: set, B: set, N: set, U: set) -> float:
    """Compute the Chaining based weight for given sets A, B, N and U.

    Args:
        A: Set A.
        B: Set B.
        N: Set of all players.
        U: Current coalition.

    Returns:
        The Chaining based weight.
    """
    u_0 = len(U.intersection(N.difference(B)))
    n = len(N)
    a = len(A)
    b = len(B)
    sign = (-1) ** (u_0)
    weight = len(U) * beta(u_0 + a, n - b - u_0 + 1)
    return sign * weight


def interaction_weight_func(
    index: str,
    index_func: Callable | None,
    p: float,
    interaction_size: int,
    coalition_size: int,
    n_players: int,
) -> float:
    """Compute the general interaction weight function.

    Args:
        index: The interaction index name.
        index_func: Custom index function if index is not recognized.
        p: Probability parameter for WBII.
        interaction_size: The coalition to compute the effect for.
        coalition_size: The coalition which is a superset of s.
        n_players: The total number of players.

    Returns:
        The interaction weight.
    """
    if index in ["SII", "SV"]:
        return 1 / (
            (n_players + interaction_size - 1) * binom(n_players - interaction_size, coalition_size)
        )
    if index in ["BII", "BV", "FBII"]:
        return 1 / (2 ** (n_players - interaction_size))
    if index in ["WBII"]:
        return (p) ** coalition_size * (1 - p) ** (n_players - interaction_size - coalition_size)
    if index in ["CHII", "CV"]:
        return interaction_size / (
            (interaction_size + coalition_size)
            * binom(n_players, coalition_size + interaction_size)
        )
    if index in ["FSII"]:
        return (
            factorial(2 * interaction_size - 1)
            / (factorial(interaction_size - 1)) ** 2
            * (
                factorial(interaction_size + coalition_size - 1)
                * factorial(n_players - coalition_size - 1)
                / factorial(n_players + interaction_size - 1)
            )
        )
    if index in ["STII"]:
        return 1 / (binom(n_players - 1, coalition_size)) * interaction_size / n_players
    warn(f"Index {index} not recognized. Checking if callable function was given.", stacklevel=2)
    if index_func is None:
        msg = f"Index function must be provided if index {index} is not recognized."
        raise ValueError(msg)
    return index_func(interaction_size, coalition_size, n_players)


def interaction_weight_to_moebius_weight(
    index: str,
    index_func: Callable | None,
    p: float,
    interaction_size: int,
    coalition_size: int,
) -> float:
    """Convert the Interaction Weight Representation to Möbius Representation.

    Args:
        index: The interaction index name.
        index_func: Custom index function if index is not recognized.
        p: Probability parameter for WBII.
        interaction_size: The coalition to compute the effect for.
        coalition_size: The coalition which is a superset of s.

    Returns:
        The Möbius weight.
    """
    return interaction_weight_func(
        index, index_func, p, interaction_size, coalition_size - interaction_size, coalition_size
    )


def interaction_weight_to_moebius_weight_gv(
    index: str,
    index_func: Callable | None,
    p: float,
    interaction_size: int,
    coalition_size: int,
    n_players: int,
) -> float:
    """Convert the Interaction Weight Representation to Möbius Representation for General Values.

    Args:
        index: The interaction index name.
        index_func: Custom index function if index is not recognized.
        p: Probability parameter for WBII.
        interaction_size: The coalition to compute the effect for.
        coalition_size: The coalition which is a superset of s.
        n_players: The total number of players.

    Returns:
        The Möbius weight for the given interaction size and coalition size.
    """
    return sum(
        [
            interaction_weight_func(
                index=index,
                index_func=index_func,
                p=p,
                interaction_size=interaction_size,
                coalition_size=level,
                n_players=level + interaction_size,
            )
            for level in range(
                n_players - coalition_size,
                n_players - interaction_size + 1,
            )
        ]
    )
