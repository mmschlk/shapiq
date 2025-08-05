

from shapiq.utils import powerset
from shapiq.approximator.sampling import CoalitionSampler
import numpy as np
from shapiq.approximator.regression.shapleygax import ExplanationBasisGenerator
from scipy.special import binom

from shapiq.games.benchmark.synthetic import SOUM

def create_kernel_weights(n_players, coalitions):
    kernel_weights = np.zeros(len(coalitions))
    random_weights = np.random.rand(n_players + 1)
    random_weights = 1 / 2 * (random_weights + random_weights[::-1])
    for i, coalition in enumerate(coalitions):
        coalition_size = np.sum(coalition)
        if coalition_size == 0 or coalition_size == n_players:
        # kernel_weights[i] = 1/2**(n_players)
            #kernel_weights[i] = 1e7
            #kernel_weights[i] = 0
            kernel_weights[i] = random_weights[coalition_size]
        else:
            kernel_weights[i] = 1 / binom(n_players - 2, coalition_size - 1)
            #kernel_weights[i] = random_weights[coalition_size]
            # kernel_weights[i] = 0.4**coalition_size* (0.6**(n_players-coalition_size))
    return kernel_weights

def create_teal_kernel_weights(n_players, coalitions):
    kernel_weights = np.zeros(len(coalitions))
    random_weights = np.random.rand(n_players+1)
    #random_weights = 1/2*(random_weights + random_weights[::-1])
    coalition_seen = {}
    random_weights = np.random.rand(len(coalitions))
    for i, coalition in enumerate(coalitions):
        coalition_size = np.sum(coalition)
        # get indices which are on of coalition store them in coalition_tuple
        coalition_tuple = tuple(np.where(coalition)[0])
        coalition_seen[coalition_tuple] = i
        complement = 1-coalition
        complement_tuple = tuple(np.where(complement)[0])
        if complement_tuple in coalition_seen:
            complement_pos = coalition_seen[complement_tuple]
            kernel_weights[i] = random_weights[complement_pos]
        else:
            kernel_weights[i] = random_weights[i]
    return kernel_weights

def create_regression_matrix(coalitions, basis):
    regression_matrix = np.zeros((len(coalitions), len(basis)))
    for coalition_pos, coalition in enumerate(coalitions):
        for interaction, interaction_pos in basis.items():
            interaction_size = len(interaction)
            intersection_size = np.sum(coalition[list(interaction)])
            regression_matrix[coalition_pos, interaction_pos] = int(
                interaction_size == intersection_size
            )
    return regression_matrix


def solve_least_squares(regression_matrix, regression_weights, regression_response):
    weighted_regression_matrix = regression_weights[:, None] * regression_matrix
    # try solving via solve function
    solution = np.linalg.solve(
        regression_matrix.T @ weighted_regression_matrix,
        weighted_regression_matrix.T @ regression_response,
    )
    return solution

def full_coalition_matrix(n_players):
    coalitions = np.zeros((2**n_players, n_players), dtype=int)
    for i,coalition in enumerate(powerset(range(n_players))):
        coalitions[i, list(coalition)] = 1
    return coalitions


def transform_interactions_to_shap(interaction_values,interaction_lookup):
    transformed_values = np.zeros_like(interaction_values)
    for interaction, interaction_pos in interaction_lookup.items():
        for i in interaction:
            transformed_values[interaction_lookup[(i,)]] += interaction_values[
                interaction_pos
            ] / len(interaction)
    # Handle the empty coalition
    transformed_values[interaction_lookup[()]] = interaction_values[interaction_lookup[()]]
    return transformed_values


def projection_matrix(regression_matrix):
    return np.linalg.inv(regression_matrix.T @ regression_matrix) @ regression_matrix.T

if __name__ == '__main__':
    n_players = 5
    #sampling_weights = np.ones(n_players+1,dtype=float)
    sampling_weights = np.random.rand(n_players+1)
    sampling_weights = 1/2*(sampling_weights+sampling_weights[::-1])
    budget = 24

    sampler = CoalitionSampler(n_players=n_players, sampling_weights=sampling_weights, random_state=42, replacement=False, pairing_trick=True)
    sampler.sample(budget)

    coalitions = sampler.coalitions_matrix

    #soum = SOUM(n=n_players, n_basis_games=10, max_interaction_size=2)

    # Exact shapley values
    #exact_shapley_values = soum.exact_values(index="SV",order=1)

    #game_values = soum(coalitions)
    # fake game values
    game_values = np.random.rand(budget)
    game_values[0] = 0.0  # Set the empty coalition value to 0

    basis_generator = ExplanationBasisGenerator(N=set(range(n_players)))
    basis_order2 = basis_generator.generate_kadd_explanation_basis(max_order=2)
    basis_order1 = basis_generator.generate_kadd_explanation_basis(max_order=1)
    basis_order3 = basis_generator.generate_kadd_explanation_basis(max_order=3)
    basis_order4 = basis_generator.generate_kadd_explanation_basis(max_order=4)
    basis_stoch = basis_generator.generate_stochastic_explanation_basis(n_explanation_terms=12)


    kernel_weights = create_kernel_weights(n_players, coalitions)
    #kernel_weights = create_teal_kernel_weights(n_players, coalitions)
    #kernel_weights = np.ones(len(coalitions))

    #kernel_weights = np.random.rand(len(coalitions))
    regression_weights = kernel_weights * sampler.sampling_adjustment_weights
    # KernelSHAP
    regression_matrix_kernelshap = create_regression_matrix(coalitions, basis_order1)
    kernelshap = solve_least_squares(regression_matrix_kernelshap, regression_weights, game_values)

    # GAX
    regression_matrix_gax2add = create_regression_matrix(coalitions, basis_order2)
    interaction_values = solve_least_squares(regression_matrix_gax2add, regression_weights, game_values)
    shapleygax2 = transform_interactions_to_shap(interaction_values, basis_order2)[:n_players+1]

    # GAX
    regression_matrix_gax3add = create_regression_matrix(coalitions, basis_order3)
    interaction_values = solve_least_squares(regression_matrix_gax3add, regression_weights, game_values)
    shapleygax3 = transform_interactions_to_shap(interaction_values, basis_order3)[:n_players+1]

    # GAX
    regression_matrix_gax4add = create_regression_matrix(coalitions, basis_order4)
    interaction_values = solve_least_squares(regression_matrix_gax4add, regression_weights, game_values)
    shapleygax4 = transform_interactions_to_shap(interaction_values, basis_order4)[:n_players+1]


    # GAX stochastic
    regression_matrix_gaxstoch = create_regression_matrix(coalitions, basis_stoch)
    interaction_values_stoch = solve_least_squares(regression_matrix_gaxstoch, regression_weights, game_values)
    shapleygax_stoch = transform_interactions_to_shap(interaction_values_stoch, basis_stoch)[:n_players+1]

    # Full
    full_coalitions = full_coalition_matrix(n_players)
    full_kernel_weights = create_kernel_weights(n_players, full_coalitions)
    full_regression_matrix = create_regression_matrix(full_coalitions, basis_order1)


    # Print results
    print("KernelSHAP values:", kernelshap)
    print("ShapleyGAX values (2add):", shapleygax2)
    print("ShapleyGAX values (3add):", shapleygax3)
    print("ShapleyGAX values (4add):", shapleygax4)
    print("ShapleyGAX stochastic values:", shapleygax_stoch)



    M = np.zeros((n_players,len(basis_order2)), dtype=float)
    for interaction, pos in basis_order2.items():
        for i in range(n_players):
            if i in interaction:
                M[i, pos] = 1/len(interaction)

    M_hat = np.linalg.inv(regression_matrix_kernelshap.T @ regression_matrix_kernelshap) @ regression_matrix_kernelshap.T
    pseudo_inverse_gax = np.linalg.pinv(regression_matrix_gax2add)

    weighted_regression_matrix_2add = np.sqrt(kernel_weights[:, None]) * regression_matrix_gax2add
    weighted_regression_matrix_3add = np.sqrt(kernel_weights[:, None]) * regression_matrix_gax3add


    samples = []
    for sample,pos in sampler.sampled_coalitions_dict.items():
        samples.append(sample)



    A = weighted_regression_matrix_2add
    AtA = A[:,:n_players+1].T @ A[:,:n_players+1]
    AtA_inv = np.linalg.pinv(AtA)
    AtB = A[:,:n_players+1].T @ A[:,n_players+1:]
    M_prime = AtA_inv @ AtB


    def row_guess(S, samples, weights):
        numerator = 0
        denominator = 0
        for sample, weight in zip(samples, weights):
            # If |S cap T| = 1
            denominator += 2 * weight
            if len(set(S).intersection(set(sample))) == 1:
                numerator -= weight
        return numerator / denominator


    def row_guess_k(S, samples, weights):
        numerator = 0
        denominator = 0
        for sample, weight in zip(samples, weights):
            denominator += weight
            for k in range(1,len(S)):
                if len(set(S).intersection(set(sample))) == k:
                    numerator -= weight * k / len(S)
        return numerator / denominator

    MAX_ORDER = 2
    Q = []
    for S in powerset(range(n_players), max_size=MAX_ORDER):
        Q.append(S)

    for S in powerset(range(n_players),max_size=MAX_ORDER):
        if len(S) >= 2:
            guess = row_guess_k(S, samples, kernel_weights)
            print(S,guess)

    buildAtAM_prime = np.zeros((n_players + 1, len(Q) - (n_players + 1)))
    for idx, S in enumerate(Q):
        if len(S) == 2:
            for i in range(n_players):
                guess = 0
                for sample, weight in zip(A, kernel_weights):
                    if i in sample:
                        guess += weight * M_prime[0, idx - (n_players + 1)]
                        for j in S:
                            if j in sample:
                                guess += weight * 1 / len(S)
                buildAtAM_prime[i + 1, idx - (n_players + 1)] = guess



