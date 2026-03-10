
from warnings import warn

import numpy as np
from scipy.special import beta, binom, factorial
from shapiq.utils.sets import powerset


def moebius_value(S, A, B):
        return sum(
            [
                (-1) ** (len(S) - len(T))
                * (1 if A.issubset(set(T)) and set(T).issubset(B) else 0)
                for T in powerset(S)
            ]
        )

def general_weight_fbii(A, B, N, U, max_order):
        """Computes the general weight for FBII for given sets A, B, N and U.

        Args:
            A: Set A.
            B: Set B.
            N: Set of all players.
            U: Current coalition.
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
        for l in range(max_order + 1 - u_0 - a, n - b - u_0 + 1):
            term = (-1) ** (u_0 + l + max_order - u) * (1 / 2) ** (a + l + u_0 - u) * binom(a + l + u_0 - u - 1, max_order - u) * binom(n - b - u_0, l)
            w += term
        return w1 + w

def shapley_weight_function(a, b):
        """Computes the Shapley weight for given set sizes a and b.

        Args:
            a: Size of set A.
            b: Size of set B.
        Returns:
            The Shapley weight.
        """
        return 1.0 / ((a + b + 1) * binom(a + b, b))
def shapley_based_weight_function(A, B, N, U):
        """Computes the Shapley based weight for given sets A, B, N and U.

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
            warn(f"Invalid sets A={A}, B={B}, N={N}, U={U}. Returning 0 instead.")
            return 0
        sign = (-1) ** (len(U.intersection(N.difference(B))))
        return sign * 1.0 / ((a + b + 1) * binom(a + b, b))

    
def banzhaf_weight_function(A, B, N, U):
        """Computes the Banzhaf based weight for given sets A, B, N and U.

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

def chaining_weight_function(A, B, N, U):
        """Computes the Chaining based weight for given sets A, B, N and U.

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


def interaction_weight_func(index, index_func, p, interaction_size, coalition_size, n_players):
        """The general API for Interaction weight functions

        Args:
            interaction_size (int): The coaltion to compute the effect for
            coalition_size (int): the coalition which is a superset of s
            n_players (int): The total number of players.
        """
        if index in ["SII", "SV"]:
            return 1 / (
                (n_players + interaction_size - 1)
                * binom(n_players - interaction_size, coalition_size)
            )
        if index in ["BII", "BV", "FBII"]:
            return 1 / (2 ** (n_players - interaction_size))
        if index in ["WBII"]:
            return (p) ** coalition_size * (1 - p) ** (
                n_players - interaction_size - coalition_size
            )
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
            return (
                1
                / (binom(n_players - 1, coalition_size))
                * interaction_size
                / n_players
            )
        warn(
            f"Index {index} not recognized. Checking if callable function was given."
        )
        if index_func is None:
            raise ValueError(
                f"Index function must be provided if index {index} is not recognized."
            )
        return index_func(interaction_size, coalition_size, n_players)

def interaction_weight_to_moebius_weight(
        index, index_func, p,
        interaction_size: int,
        coalition_size: int,
    ):
        """Converts the Interaction Weight Representation to Möbius Representation.

        Args:
            interaction_size (int): The coalition to compute the effect for
            coalition_size (int): The coalition which is a superset of s.
            interaction_weight_func (N x N -> R): The interaction weight function.
        """
        return interaction_weight_func(
            index, index_func, p, interaction_size, coalition_size - interaction_size, coalition_size
        )

def interaction_weight_to_moebius_weight_gv(
        index, index_func, p,
        interaction_size: int,
        coalition_size: int,
        n_players: int,
    ):
        return sum(
            [
                interaction_weight_func(
                    index=index,
                    index_func=index_func,
                    p=p,
                    interaction_size=interaction_size,
                    coalition_size=l,
                    n_players=l + interaction_size,
                )
                for l in range(
                    n_players - coalition_size,
                    n_players - interaction_size + 1,
                )
            ]
        )
