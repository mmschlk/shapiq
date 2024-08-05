from scipy.optimize import linprog,minimize, LinearConstraint
import numpy as np

def setup_core_calculations(coalition_values, coalition_matrix, positive_contraint=False,
                            fixed_subsidy = None):
    """
    Converts the coalition_values and coalition_matrix into a linear programming problem using scipy.linprog.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html for reference.

    Args:
        coalition_values:
        coalition_matrix: binary matrix with shape (n_coaltions, n_players) where entry (i,j) denotes that in coaltion i the player j is present
        positive_constraint: Indicates whether we want the credit_assignments only to have positive values. Defaults to False.
        fixed_subsidy: The value we want the subsidy to have. None means the subsidy is also optimized. Defaults to None.
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

    # Setup player credit assignment bounds
    if positive_contraint:
        bounds_players = [(0,None) for _ in range(n_players)]
    else:
        # Bounds for the values of payoff and e
        bounds_players = [(None, None) for _ in range(n_players)]

    bounds_players += [(fixed_subsidy, fixed_subsidy)]


    # Minimizer Coefficients
    c = np.ones(n_players + 1)

    # Convert the Constraints to the form for egalitarian least-core optimization
    credit_assignment_constraints = LinearConstraint(A_ub,ub=b_ub)
    efficiency_constraint = LinearConstraint(A_eq,lb=b_eq,ub=b_eq)

    constraints = [credit_assignment_constraints, efficiency_constraint]

    return constraints, bounds_players

def minimization_egal_least_core(credit_subsidy_vector):
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
    return np.linalg.norm(credit_assignment,ord=2) + subsidy
def solve_egalitarian_e_core(coalition_values, coalition_matrix, positive_constraint=False, e=None):
    constraints, bounds = setup_core_calculations(coalition_values,
                                                           coalition_matrix,
                                                           positive_constraint,
                                                           fixed_subsidy=e)

    res = minimize(fun=minimization_egal_least_core,
                   x0=np.zeros(coalition_matrix.shape[1]+1),
                   bounds=bounds,
                   constraints=constraints)
    if res.success:
        return res.x[:-1],res.x[-1]
    else:
        raise ValueError("A solution was not found for the given game and parameters!")

def get_egalitarian_least_core(coaliton_values,coalition_matrix,positive_constraint=False):
    credit_assignment,subsidy = solve_egalitarian_e_core(coaliton_values,coalition_matrix,positive_constraint,e=None)

    return (credit_assignment,subsidy)

def get_egalitarian_core(coaliton_values,coalition_matrix,positive_constraint=False):
    credit_assignment,subsidy = solve_egalitarian_e_core(coaliton_values,coalition_matrix,positive_constraint,e=0)

    return (credit_assignment,subsidy)


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


credit_assignment, e = get_egalitarian_least_core(coalition_values, coalition_matrix,
                                                positive_constraint=False)
print("Credit Assignment: ", credit_assignment)
print("Subsidy: ", e)
