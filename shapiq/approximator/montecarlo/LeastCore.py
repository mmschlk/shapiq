from scipy.optimize import linprog
import numpy as np

def setup_core_calculations(coalition_values, coalition_matrix):
    """
    Converts the coalition_values and coalition_matrix into a linear programming problem using scipy.linprog.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html for reference.

    Args:
        coalition_values:
        coalition_matrix: binary matrix with shape (n_coaltions, n_players) where entry (i,j) denotes that in coaltion i the player j is present

    Returns:
        (c,A_ub,b_ub,A_eq,b_eq, bounds) where
           min (x,e) in c @ (x,e)
           such that
                A_ub @ x <= b_ub
                A_eq @ x == b_eq
                x in bounds
    """
    n_coalitions, n_players = coalition_matrix.shape

    grand_coaltion_value = coalition_values[
        tuple(np.where(coalition_matrix[-1])[0])
    ]

    # Setup the binary matrix representing the linear inequalities for core
    A_ub = np.ones((n_coalitions - 1, n_players + 1))
    A_ub[:, :-1] = coalition_matrix[:-1]
    A_ub[0, -1] = 0

    # Setup the upper bounds for the inequalities (negative coalition values)

    b_ub = (-1) * np.array([
        coalition_values[tuple(
            np.where(x)[0]  # Convert binary matrix into tuple to index into coalition values
        )] for x in coalition_matrix
        if np.sum(x) < n_players  # The grandCoalition is not an inequality but an equality
    ])

    # Due to the Core  consisting of (>=) we need the (-1) to have (<=).
    A_ub *= (-1)

    # Setup the binary matrix representing the efficiency property
    A_eq = np.ones((1, n_players + 1))

    # Let the e not be contained in the efficiency property
    A_eq[0, -1] = 0

    # Efficiency value
    b_eq = np.array([grand_coaltion_value])

    # Bounds for the values of payoff and e
    bounds_players = [(None, None) for _ in range(n_players + 1)]

    # Minimizer Coefficients
    c = np.ones(n_players + 1)

    return c,A_ub,b_ub,A_eq,b_eq, bounds_players

def solve_e_Core(coalition_values, coalition_matrix, e):
    """
    Solves for the e-Core given coalition_values and coalition_matrix when having fixed external subsidy e.
    Args:
        coalition_values:
        coalition_matrix: binary matrix with shape (n_coaltions, n_players) where entry (i,j) denotes that in coaltion i the player j is present
        e: External subsidy provided in the Game

    Returns:
        (x,e) where x is the credit assignment of the players and e is the external subsidy.

    """
    pass
def solve_least_core(coalition_values, coalition_matrix):
    """
        Solves for the Least-Core given coaltion_values and coalition_matrix.
    Args:
        coalition_values:
        coalition_matrix: binary matrix with shape (n_coaltions, n_players) where entry (i,j) denotes that in coaltion i the player j is present

    Returns:
        (x,e) where x is the credit assignment of the players and e is the external subsidy.

    """
    c,A_ub,b_ub,A_eq,b_eq,bounds = setup_core_calculations(coalition_values, coalition_matrix)

    res = linprog(c=c,
                  A_ub=A_ub,
                  b_ub=b_ub,
                  A_eq=A_eq,
                  b_eq=b_eq,
                  bounds=bounds)


    return (res.x[:-1],res.x[-1])

coalition_values = {
    ():0,
    (0,):0,
    (1,):0,
    (2,):0,
    (0,1):100,
    (0,2):80,
    (1,2):70,
    (0,1,2):100
}
coalition_matrix = np.array([[0,0,0],
                             [1,0,0],
                             [0,1,0],
                             [0,0,1],
                             [1,1,0],
                             [1,0,1],
                             [0,1,1],
                             [1,1,1]])


credit_assignment, e = solve_least_core(coalition_values,coalition_matrix)
print("Credit Assignment: ", credit_assignment)
print("Subsidy: ", e)