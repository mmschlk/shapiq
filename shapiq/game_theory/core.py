"""Logic to solve for the egalitarian least-core."""

from __future__ import annotations

import copy
import warnings

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

__all__ = ["egalitarian_least_core"]


def _setup_core_calculations(
    n_players: int,
    game_values: np.ndarray,
) -> tuple[list[LinearConstraint], list[tuple[int | None, int | None]]]:
    """Setup core optimization matrices for scipy.linprog.

    Args:
        n_players: amount of players in the game
        game_values: the values of every coalition in the game. Assumes empty set in game_values[0] and grand_coalition
            game_values[-1]

    Returns:
        (constraints,bounds): returns the constraints and bounds induced by the stability and efficiency property
         for the underlying game.

    """
    n_coalitions = 2**n_players

    coalition_matrix = np.zeros((n_coalitions, n_players), dtype=int)
    for i, T in enumerate(powerset(set(range(n_players)), min_size=0, max_size=n_players)):
        coalition_matrix[i, T] = 1  # one-hot-encode the coalition

    grand_coaltion_value = game_values[-1]

    # Setup the binary matrix representing the linear inequalities for core  except for the grand coalition
    stability_matrix = np.ones(
        (n_coalitions - 1, n_players + 1),
    )  # $A_\{ub\}$. Optimization upper bound values.
    stability_matrix[:, :-1] = coalition_matrix[:-1]
    stability_matrix[0, -1] = 0

    # Due to scipy.optimize we need to convert the stability inequality (>=) to (<=) via (-1)
    stability_matrix *= -1

    # Setup the upper bounds for the inequalities (negative coalition values).
    # The grand_coalition value (game_values[-1]) is hereby excluded
    stability_values = (-1) * game_values[:-1]

    # Setup the binary matrix representing the efficiency property
    efficiency_matrix = np.ones((1, n_players + 1))  # $A_\{eq\}$. Optimization equality values.

    # Let the e not be contained in the efficiency property
    efficiency_matrix[0, -1] = 0

    # Efficiency value
    efficiency_value = np.array([grand_coaltion_value])  # $b_\{eq\}$.

    # Bounds for the values of credit_assignments
    bounds_players = [(None, None) for _ in range(n_players)]

    # Bounds for the subsidy
    bounds_players += [(0, None)]

    # Convert the Constraints to the form for egalitarian least-core optimization
    # $A_\{ub\}$ @ (x,e) <= $b_\{ub\}$
    credit_assignment_constraints = LinearConstraint(stability_matrix, ub=stability_values)
    # $A_\{eq\} @ (x,e) == $b_\{eq\}$
    efficiency_constraint = LinearConstraint(
        efficiency_matrix,
        lb=efficiency_value,
        ub=efficiency_value,
    )

    constraints = [credit_assignment_constraints, efficiency_constraint]

    return constraints, bounds_players


def _minimization_egal_least_core(credit_subsidy_vector: np.ndarray) -> float:
    """Formulates the minimization problem to find the egalitarian least-core given the guess credit_subsidy_vector.

    Args:
        credit_subsidy_vector: ndarray with shape (n_players + 1,) where the last element is the external subsidy e.

    Returns:
        A value representing the sum of both l2_norm of the credit_assignment and subsidy.

    """
    credit_assignment = credit_subsidy_vector[:-1]
    subsidy = credit_subsidy_vector[-1]
    # Computes the egalitarian_least_core value and e
    return np.linalg.norm(credit_assignment, ord=2) + subsidy


def egalitarian_least_core(
    n_players: int,
    game_values: np.ndarray,
    coalition_lookup: dict[tuple[int], int],
) -> tuple[InteractionValues, float]:
    """Computes the egalitarian least-core for the underlying game represented through the parameters.

    Args:
        n_players: amount of players in the game.
        game_values: the values of every coalition in the game.
        coalition_lookup: dictionary mapping a coalition to the corresponding value of game_values.

    Returns:
        Returns a tuple of egalitarian_least_core and subsidy value.

    Raises:
        ValueError: If the optimization did not complete successfully

    """
    player_set = set(range(n_players))

    # Rearrange the game_values and base_line and 0
    tmp = game_values[coalition_lookup[()]]
    game_values[coalition_lookup[()]] = game_values[0]
    game_values[0] = tmp

    # Rearrange the game_values to have grand_coalition at -1
    tmp = game_values[coalition_lookup[tuple(range(n_players))]]
    game_values[coalition_lookup[tuple(range(n_players))]] = game_values[-1]
    game_values[-1] = tmp

    baseline_value = game_values[0]

    # Check for normalized game
    if baseline_value != 0:
        # Normalize the game for the ELC computation
        warnings.warn(
            "The egalitarian least core is only defined for normalized games."
            "Thus the resulting vector will undercut efficiency by the value of the empty set."
            "To suppress warnings normalize the game to have baseline_value == 0.",
            stacklevel=2,
        )

    # Potentially normalize the game
    game_values = game_values - baseline_value

    constraints, bounds = _setup_core_calculations(n_players, game_values)

    # Find egalitarian_least_core with subsidy
    res = minimize(
        fun=_minimization_egal_least_core,
        x0=np.zeros(n_players + 1),
        bounds=bounds,
        constraints=constraints,
    )

    # Build interaction_lookup for plotting functions
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(player_set, min_size=1, max_size=1))
    }

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
