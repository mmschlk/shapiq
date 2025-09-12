from __future__ import annotations

from scipy.special import binom

from shapiq.approximator.regression.shapleygax import ExplanationBasisGenerator

if __name__ == "__main__":
    # Example usage of ExplanationBasisGenerator
    n_players = 8
    N = set(range(n_players))
    random_state = 42

    basis_generator = ExplanationBasisGenerator(N)

    # Generate the explanation basis
    explanation_basis = basis_generator.generate_partial_explanation_basis(
        n_explanation_terms=n_players + 1
    )
    print("Order 1: ", explanation_basis)

    ratios = [0.1, 0.2, 0.5, 1]
    for ratio in ratios:
        n_explanations = 1 + n_players + ratio * int(binom(n_players, 2))
        explanation_basis = basis_generator.generate_partial_explanation_basis(
            n_explanation_terms=n_explanations
        )
        print("Order 2:", explanation_basis)
        explanation_basis = basis_generator.generate_stochastic_explanation_basis(
            n_explanation_terms=n_explanations
        )
        print("Stochastic Order 2:", explanation_basis)
        n_explanations_3 = 1 + n_players + int(ratio * binom(n_players, 3))
        explanation_basis = basis_generator.generate_partial_explanation_basis(
            n_explanation_terms=n_explanations_3, sizes_to_exclude=[2]
        )
        print("Order 3:", explanation_basis)
