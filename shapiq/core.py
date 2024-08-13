"""Logic to solve for the egalitarian least-core."""

import copy
from typing import Optional

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

__all__ = ["egalitarian_least_core"]


def _setup_core_calculations(
    grand_coalition_tuple: tuple[int],
    n_players: int,
    game_values: np.ndarray,
    coalition_lookup: dict[tuple[int], int],
) -> tuple[list[LinearConstraint], list[tuple[Optional[int], Optional[int]]]]:
    """
    Converts the coalition_values and coalition_matrix into a linear programming problem using scipy.linprog.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html for reference.

    Args:
        grand_coalition_tuple: tuple representing the grand_coalition of the underlying game
        n_players: amount of players in the game
        game_values: the values of every coalition in the game
        coalition_lookup: dictionary mapping a coalition to the corresponding value of game_values.
    Returns:
        (constraints,bounds): returns the constraints and bounds induced by the stability and efficiency property
         for the underlying game.
    """
    n_coalitions = 2**n_players

    coalition_matrix = np.zeros((n_coalitions, n_players), dtype=int)
    for i, T in enumerate(powerset(set(grand_coalition_tuple), min_size=0, max_size=n_players)):
        coalition_matrix[i, T] = 1  # one-hot-encode the coalition

    grand_coaltion_value = game_values[coalition_lookup[grand_coalition_tuple]]

    # Setup the binary matrix representing the linear inequalities for core  except for the grand coalition
    A_ub = np.ones((n_coalitions - 1, n_players + 1))
    A_ub[:, :-1] = coalition_matrix[:-1]
    A_ub[0, -1] = 0

    # Due to scipy.optimize we need to convert the stability inequality (>=) to (<=) via (-1)
    A_ub *= -1

    # Setup the upper bounds for the inequalities (negative coalition values).
    # The grand_coalition value (game_values[-1]) is hereby excluded
    b_ub = (-1) * game_values[:-1]

    # Setup the binary matrix representing the efficiency property
    A_eq = np.ones((1, n_players + 1))

    # Let the e not be contained in the efficiency property
    A_eq[0, -1] = 0

    # Efficiency value
    b_eq = np.array([grand_coaltion_value])

    # Bounds for the values of credit_assignments
    bounds_players = [(None, None) for _ in range(n_players)]

    # Bounds for the subsidy
    bounds_players += [(0, None)]

    # Convert the Constraints to the form for egalitarian least-core optimization
    # A_ub @ (x,e) <= b_ub
    credit_assignment_constraints = LinearConstraint(A_ub, ub=b_ub)
    # A_eq @ (x,e) == b_eq
    efficiency_constraint = LinearConstraint(A_eq, lb=b_eq, ub=b_eq)

    constraints = [credit_assignment_constraints, efficiency_constraint]

    return constraints, bounds_players


def _minimization_egal_least_core(credit_subsidy_vector: np.ndarray) -> float:
    """
    Formulates the minimization problem to find the egalitarian least-core given the guess credit_subsidy_vector.

    Args:
        credit_subsidy_vector: ndarray with shape (n_playes+1,) where the last element is the external subsidy e.

    Returns:
        A value representing the sum of both l2_norm of the credit_assignment and subsidy.
    """
    credit_assignment = credit_subsidy_vector[:-1]
    subsidy = credit_subsidy_vector[-1]
    # Computes the egalitarian_least_core value and e
    return np.linalg.norm(credit_assignment, ord=2) + subsidy


def egalitarian_least_core(
    grand_coalition_tuple: tuple[int],
    n_players: int,
    game_values: np.ndarray,
    coalition_lookup: dict[tuple[int], int],
) -> tuple[InteractionValues, float]:
    """
    Computes the egalitarian least-core for the underlying game represented through the parameters.
    Args:
        grand_coalition_tuple: tuple representing the grand_coalition of the underlying game
        n_players: amount of players in the game
        game_values: the values of every coalition in the game
        coalition_lookup: dictionary mapping a coalition to the corresponding value of game_values.
    Returns:
        (egalitarian_least_core, subsidy): Returns the optimization result for the underlying game.
        Meaning the egalitarian_least_core is a stable payoff given the subsidy.

    Raises:
        ValueError: If the optimization did not complete successfully
    """

    constraints, bounds = _setup_core_calculations(
        grand_coalition_tuple, n_players, game_values, coalition_lookup
    )

    # Find egalitarian_least_core with subsidy
    res = minimize(
        fun=_minimization_egal_least_core,
        x0=np.zeros(n_players + 1),
        bounds=bounds,
        constraints=constraints,
    )

    # Check if optimization was successfull
    if not res.success:
        raise ValueError("A stable credit assignment was not found in the game !")

    # Build interaction_lookup for plotting functions
    interaction_lookup = {}
    for i, interaction in enumerate(powerset(set(grand_coalition_tuple), min_size=1, max_size=1)):
        interaction_lookup[interaction] = i

    credit_assignment, subsidy = res.x[:-1], res.x[-1]

    # Create InteractionValues
    egalitarian_least_core = InteractionValues(
        values=credit_assignment,
        index="ELC",
        max_order=1,
        min_order=1,
        n_players=n_players,
        interaction_lookup=interaction_lookup,
        estimated=False,
        baseline_value=0,
    )

    # Check subsidy close to zero
    if subsidy < 10e-7:
        subsidy = 0

    return copy.copy(egalitarian_least_core), subsidy
