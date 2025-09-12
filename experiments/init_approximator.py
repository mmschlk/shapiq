from __future__ import annotations

import numpy as np
from scipy.special import binom

from shapiq import PermutationSamplingSV
from shapiq.approximator.regression.shapleygax import (
    ExplanationBasisGenerator,
    ShapleyGAX,
)
from shapiq.utils.empirical_leverage_scores import get_leverage_scores


def get_approximators(
    APPROXIMATORS, n_players, RANDOM_STATE, PAIRING, REPLACEMENT, FORCE_BORDERS=False
):
    # Create the approximators for the game

    # initialize the weights for KernelSHAP
    kernelshap_weights = np.zeros(n_players + 1)
    for size in range(1, n_players):
        kernelshap_weights[size] = 1 / (size * (n_players - size))

    # initialize the weights for LeverageSHAP
    leverage_weights_1 = np.ones(n_players + 1)

    # initialize the weights for LeverageSHAP order 2
    lev_scores_2 = get_leverage_scores(n_players, 2)
    leverage_weights_2 = np.zeros(n_players + 1)
    for size, score in lev_scores_2.items():
        leverage_weights_2[size] = binom(n_players, size) * score

    approximators = []

    explanation_basis = ExplanationBasisGenerator(N=set(range(n_players)))
    kadd1 = explanation_basis.generate_kadd_explanation_basis(max_order=1)
    kadd2 = explanation_basis.generate_kadd_explanation_basis(max_order=2)
    kadd3 = explanation_basis.generate_kadd_explanation_basis(max_order=3)
    kadd4 = explanation_basis.generate_kadd_explanation_basis(max_order=4)

    ksym1 = explanation_basis.generate_ksym_explanation_basis(max_order=1)
    ksym2 = explanation_basis.generate_ksym_explanation_basis(max_order=2)

    if "KernelSHAP" in APPROXIMATORS:
        # KernelSHAP
        # kernel_shap = KernelSHAP(n=n_players, random_state=RANDOM_STATE, sampling_weights=kernelshap_weights, pairing_trick=PAIRING, replacement=REPLACEMENT)
        kernel_shap = ShapleyGAX(
            n_players,
            explanation_basis=kadd1,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        kernel_shap.name = "KernelSHAP"
        approximators.append(kernel_shap)
    if "LeverageSHAP" in APPROXIMATORS:
        # LeverageSHAP
        # leverage_shap = KernelSHAP(n=n_players, random_state=RANDOM_STATE, sampling_weights=leverage_weights_1, pairing_trick=PAIRING, replacement=REPLACEMENT)
        leverage_shap = ShapleyGAX(
            n_players,
            explanation_basis=kadd1,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        leverage_shap.name = "LeverageSHAP"
        approximators.append(leverage_shap)
    if "PermutationSampling" in APPROXIMATORS:
        # Permutation Sampling
        permutation_sampling = PermutationSamplingSV(n=n_players, random_state=RANDOM_STATE)
        permutation_sampling.name = "PermutationSampling"
        approximators.append(permutation_sampling)
    if "ShapleyGAX-1SYM-Lev1" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=ksym1,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-1SYM-Lev1"
        approximators.append(shapley_gax)
    if "ShapleyGAX-2SYM-Lev1" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=ksym2,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-2SYM-Lev1"
        approximators.append(shapley_gax)
    if "ShapleyGAX-2ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        shapley_gax_kadd = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd2,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_kadd.name = "ShapleyGAX-2ADD"
        approximators.append(shapley_gax_kadd)
    if "ShapleyGAX-2ADD-Lev1" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 1
        shapley_gax_kadd_lev1 = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd2,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_kadd_lev1.name = "ShapleyGAX-2ADD-Lev1"
        approximators.append(shapley_gax_kadd_lev1)
    if "ShapleyGAX-2ADD-Lev2" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_kadd_lev2 = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd2,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_2,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_kadd_lev2.name = "ShapleyGAX-2ADD-Leverage2"
        approximators.append(shapley_gax_kadd_lev2)
    if "ShapleyGAX-3ADD" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_3add = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd3,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_3add.name = "ShapleyGAX-3ADD"
        approximators.append(shapley_gax_3add)
    if "ShapleyGAX-3ADD-Lev1" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_add3lev = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd3,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_add3lev.name = "ShapleyGAX-3ADD-Lev1"
        approximators.append(shapley_gax_add3lev)
    if "ShapleyGAX-4ADD" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_4add = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd4,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_4add.name = "ShapleyGAX-4ADD"
        approximators.append(shapley_gax_4add)
    if "ShapleyGAX-4ADD-Lev1" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_add4lev = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd4,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_add4lev.name = "ShapleyGAX-4ADD-Lev1"
        approximators.append(shapley_gax_add4lev)
    if "ShapleyGAX-3ADD-Lev2" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_add3lev2 = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd3,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_2,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_add3lev2.name = "ShapleyGAX-3ADD-Lev1"
        approximators.append(shapley_gax_add3lev2)
    if "ShapleyGAX-3ADD-WO2" in APPROXIMATORS:
        kadd3wo2 = explanation_basis.generate_kadd_explanation_basis(
            max_order=3, sizes_to_exclude=[2]
        )
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_add3wo2 = ShapleyGAX(
            n=n_players,
            explanation_basis=kadd3wo2,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_add3wo2.name = "ShapleyGAX-3ADD-WO2"
        approximators.append(shapley_gax_add3wo2)
    if "ShapleyGAX-3ADDWO2-10%" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        n_explanations = 1 + n_players + int(0.1 * binom(n_players, 3))
        basis = explanation_basis.generate_stochastic_explanation_basis(
            n_explanation_terms=n_explanations, sizes_to_exclude=[2]
        )
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADDWO2-10%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADDWO2-20%" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        n_explanations = 1 + n_players + int(0.2 * binom(n_players, 3))
        basis = explanation_basis.generate_stochastic_explanation_basis(
            n_explanation_terms=n_explanations, sizes_to_exclude=[2]
        )
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADDWO2-20%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADDWO2-10%" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        n_explanations = 1 + n_players + int(0.5 * binom(n_players, 3))
        basis = explanation_basis.generate_stochastic_explanation_basis(
            n_explanation_terms=n_explanations, sizes_to_exclude=[2]
        )
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=kernelshap_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADDWO2-50%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-2ADD-10%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.1 * binom(n_players, 2))
        basis = explanation_basis.generate_stochastic_explanation_basis(
            n_explanation_terms=n_explanations
        )
        # ShapleyGAX with leverage weights for order 1
        shapley_gax_kadd_10p = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_kadd_10p.name = "ShapleyGAX-2ADD-10%"
        approximators.append(shapley_gax_kadd_10p)
    if "ShapleyGAX-2ADD-20%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.2 * binom(n_players, 2))
        basis = explanation_basis.generate_stochastic_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax_kadd_20p = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_kadd_20p.name = "ShapleyGAX-2ADD-20%"
        approximators.append(shapley_gax_kadd_20p)
    if "ShapleyGAX-2ADD-50%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.5 * binom(n_players, 2))
        basis = explanation_basis.generate_stochastic_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax_kadd_50p = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax_kadd_50p.name = "ShapleyGAX-2ADD-50%"
        approximators.append(shapley_gax_kadd_50p)
    if "ShapleyGAX-3ADD-10%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2)) + int(0.10 * binom(n_players, 3))
        basis = explanation_basis.generate_stochastic_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADD-10%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADD-20%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2)) + int(0.20 * binom(n_players, 3))
        basis = explanation_basis.generate_stochastic_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADD-20%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADD-50%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2)) + int(0.50 * binom(n_players, 3))
        basis = explanation_basis.generate_stochastic_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADD-50%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADDWO2-P10%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.10 * binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(
            n_explanations, sizes_to_exclude=[2]
        )
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADDWO2-P10%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADDWO2-P20%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.20 * binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(
            n_explanations, sizes_to_exclude=[2]
        )
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADDWO2-P20%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADDWO2-P50%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.50 * binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(
            n_explanations, sizes_to_exclude=[2]
        )
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADDWO2-P50%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADDWO2-P100%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(
            n_explanations, sizes_to_exclude=[2]
        )
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADDWO2-P100%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADD-P10%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2)) + int(0.10 * binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADD-P10%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADD-P20%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2)) + int(0.2 * binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADD-P20%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADD-P50%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2)) + int(0.5 * binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADD-P50%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-3ADD-P100%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2)) + int(binom(n_players, 3))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-3ADD-P100%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-2ADD-P10%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.1 * binom(n_players, 2))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-2ADD-P10%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-2ADD-P20%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.2 * binom(n_players, 2))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-2ADD-P20%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-2ADD-P50%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(0.5 * binom(n_players, 2))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-2ADD-P50%"
        approximators.append(shapley_gax)
    if "ShapleyGAX-2ADD-P100%" in APPROXIMATORS:
        n_explanations = 1 + n_players + int(binom(n_players, 2))
        basis = explanation_basis.generate_partial_explanation_basis(n_explanations)
        # ShapleyGAX with leverage weights for order 1
        shapley_gax = ShapleyGAX(
            n=n_players,
            explanation_basis=basis,
            random_state=RANDOM_STATE,
            sampling_weights=leverage_weights_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            force_borders=FORCE_BORDERS,
        )
        shapley_gax.name = "ShapleyGAX-2ADD-P100%"
        approximators.append(shapley_gax)
    return approximators
