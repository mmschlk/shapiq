
from shapiq import KernelSHAP, PermutationSamplingSV, SPEX
from shapiq.approximator.regression.shapleygax import ShapleyGAX, ExplanationBasisGenerator

from shapiq.utils.empirical_leverage_scores import get_leverage_scores

from scipy.special import binom
import numpy as np

def get_approximators(APPROXIMATORS,n_players,RANDOM_STATE,PAIRING,REPLACEMENT):
    # Create the approximators for the game


    # initialize the weights for KernelSHAP
    kernelshap_weights = np.zeros(n_players + 1)
    for size in range(1,n_players):
        kernelshap_weights[size] = 1/(size*(n_players - size))

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
    kadd3wo2 = explanation_basis.generate_kadd_explanation_basis(max_order=3,sizes_to_exclude=[2])
    if "KernelSHAP" in APPROXIMATORS:
        # KernelSHAP
        #kernel_shap = KernelSHAP(n=n_players, random_state=RANDOM_STATE, sampling_weights=kernelshap_weights, pairing_trick=PAIRING, replacement=REPLACEMENT)
        kernel_shap = ShapleyGAX(n_players, explanation_basis=kadd1, random_state=RANDOM_STATE, sampling_weights=kernelshap_weights, pairing_trick=PAIRING, replacement=REPLACEMENT)
        kernel_shap.name = "KernelSHAP"
        approximators.append(kernel_shap)
    if "LeverageSHAP" in APPROXIMATORS:
        # LeverageSHAP
        #leverage_shap = KernelSHAP(n=n_players, random_state=RANDOM_STATE, sampling_weights=leverage_weights_1, pairing_trick=PAIRING, replacement=REPLACEMENT)
        leverage_shap = ShapleyGAX(n_players, explanation_basis=kadd1, random_state=leverage_weights, sampling_weights=kernelshap_weights, pairing_trick=PAIRING, replacement=REPLACEMENT)
        leverage_shap.name = "LeverageSHAP"
        approximators.append(leverage_shap)
    if "PermutationSampling" in APPROXIMATORS:
        # Permutation Sampling
        permutation_sampling = PermutationSamplingSV(n=n_players, random_state=RANDOM_STATE)
        permutation_sampling.name = "PermutationSampling"
        approximators.append(permutation_sampling)
    if "SPEX" in APPROXIMATORS:
        # SPEX
        spex = SPEX(n=n_players, random_state=RANDOM_STATE)
        spex.name = "SPEX"
        approximators.append(spex)
    if "ShapleyGAX-2ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        shapley_gax_kadd = ShapleyGAX(n=n_players, explanation_basis=kadd2, random_state=RANDOM_STATE, sampling_weights=kernelshap_weights,pairing_trick=PAIRING,replacement=REPLACEMENT)
        shapley_gax_kadd.name = "ShapleyGAX-2ADD"
        approximators.append(shapley_gax_kadd)
    if "ShapleyGAX-2ADD-Lev1" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 1
        shapley_gax_kadd_lev1 = ShapleyGAX(n=n_players, explanation_basis=kadd2, random_state=RANDOM_STATE,
                                      sampling_weights=leverage_weights_1, pairing_trick=PAIRING, replacement=REPLACEMENT)
        shapley_gax_kadd_lev1.name = "ShapleyGAX-2ADD-Lev1"
        approximators.append(shapley_gax_kadd_lev1)
    if "ShapleyGAX-2ADD-Lev2" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_kadd_lev2 = ShapleyGAX(n=n_players, explanation_basis=kadd2, random_state=RANDOM_STATE,
                                           sampling_weights=leverage_weights_2, pairing_trick=PAIRING, replacement=REPLACEMENT)
        shapley_gax_kadd_lev2.name = "ShapleyGAX-2ADD-Leverage2"
        approximators.append(shapley_gax_kadd_lev2)
    if "ShapleyGAX-3ADD" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_3add = ShapleyGAX(n=n_players, explanation_basis=kadd3, random_state=RANDOM_STATE,
                                           sampling_weights=kernelshap_weights, pairing_trick=PAIRING, replacement=REPLACEMENT)
        shapley_gax_3add.name = "ShapleyGAX-3ADD"
        approximators.append(shapley_gax_3add)
    if "ShapleyGAX-3ADD-Lev1" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_add3lev = ShapleyGAX(n=n_players, explanation_basis=kadd3, random_state=RANDOM_STATE,
                                      sampling_weights=leverage_weights_1, pairing_trick=PAIRING,
                                      replacement=REPLACEMENT)
        shapley_gax_add3lev.name = "ShapleyGAX-3ADD-Lev1"
        approximators.append(shapley_gax_add3lev)
    if "ShapleyGAX-3ADD-Lev2" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_add3lev2 = ShapleyGAX(n=n_players, explanation_basis=kadd3, random_state=RANDOM_STATE,
                                      sampling_weights=leverage_weights_2, pairing_trick=PAIRING,
                                      replacement=REPLACEMENT)
        shapley_gax_add3lev2.name = "ShapleyGAX-3ADD-Lev1"
        approximators.append(shapley_gax_add3lev2)
    if "ShapleyGAX-3ADD-WO2" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_add3wo2 = ShapleyGAX(n=n_players, explanation_basis=kadd3wo2, random_state=RANDOM_STATE,
                                      sampling_weights=kernelshap_weights, pairing_trick=PAIRING,
                                      replacement=REPLACEMENT)
        shapley_gax_add3wo2.name = "ShapleyGAX-3ADD-WO2"
        approximators.append(shapley_gax_add3wo2)
    return approximators
