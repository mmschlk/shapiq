from __future__ import annotations

import numpy as np
from scipy.special import binom, comb

from shapiq import PermutationSamplingSV, UnbiasedKernelSHAP, SVARM
from shapiq.approximator.regressionMSR import RegressionMSR
from shapiq.approximator.regression.polyshap import (
    ExplanationFrontierGenerator,
    PolySHAP,
)
from shapiq.utils.empirical_leverage_scores import get_leverage_scores


def get_approximators(APPROXIMATORS, n_players, RANDOM_STATE, PAIRING, REPLACEMENT):
    # Create the approximators for the game

    # initialize the weights for LeverageSHAP Order 1
    sampling_weights_leverage_1 = np.ones(n_players + 1)

    approximator_list = []

    frontier_generator = ExplanationFrontierGenerator(N=set(range(n_players)))

    if "KernelSHAP" in APPROXIMATORS:
        # initialize the weights for KernelSHAP
        sampling_weights_kernelshap = np.zeros(n_players + 1)
        for size in range(1, n_players):
            sampling_weights_kernelshap[size] = 1 / (size * (n_players - size))

        explanation_frontier = frontier_generator.generate_kadd(max_order=1)
        kernel_shap = PolySHAP(
            n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_kernelshap,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        kernel_shap.name = "KernelSHAP"
        approximator_list.append(kernel_shap)

    if "LeverageSHAP" in APPROXIMATORS:
        # LeverageSHAP
        explanation_frontier = frontier_generator.generate_kadd(max_order=1)
        leverage_shap = PolySHAP(
            n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        leverage_shap.name = "LeverageSHAP"
        approximator_list.append(leverage_shap)

    if "PolySHAP-2ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        explanation_frontier = frontier_generator.generate_kadd(max_order=2)
        polyshap_2add = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_2add.name = "PolySHAP-2ADD"
        approximator_list.append(polyshap_2add)

    if "PolySHAP-3ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        explanation_frontier = frontier_generator.generate_kadd(max_order=3)
        polyshap_3add = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add.name = "PolySHAP-3ADD"
        approximator_list.append(polyshap_3add)

    if "PolySHAP-4ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        explanation_frontier = frontier_generator.generate_kadd(max_order=4)
        polyshap_4add = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_4add.name = "PolySHAP-4ADD"
        approximator_list.append(polyshap_4add)

    if "PolySHAP-2ADD-10%" in APPROXIMATORS:
        n_coefficients = 1 + n_players + int(0.1 * binom(n_players, 2))
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_2add_10 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_2add_10.name = "PolySHAP-2ADD-10%"
        approximator_list.append(polyshap_2add_10)

    if "PolySHAP-2ADD-20%" in APPROXIMATORS:
        n_coefficients = 1 + n_players + int(0.2 * binom(n_players, 2))
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_2add_20 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_2add_20.name = "PolySHAP-2ADD-20%"
        approximator_list.append(polyshap_2add_20)

    if "PolySHAP-2ADD-50%" in APPROXIMATORS:
        n_coefficients = 1 + n_players + int(0.5 * binom(n_players, 2))
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_2add_50 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_2add_50.name = "PolySHAP-2ADD-50%"
        approximator_list.append(polyshap_2add_50)

    if "PolySHAP-2ADD-75%" in APPROXIMATORS:
        n_coefficients = 1 + n_players + int(0.75 * binom(n_players, 2))
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_2add_75 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_2add_75.name = "PolySHAP-2ADD-75%"
        approximator_list.append(polyshap_2add_75)
    if "PolySHAP-3ADD-10%" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(0.1 * comb(n_players, 3))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_10 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_10.name = "PolySHAP-3ADD-10%"
        approximator_list.append(polyshap_3add_10)
    if "PolySHAP-3ADD-20%" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(0.2 * comb(n_players, 3))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_20 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_20.name = "PolySHAP-3ADD-20%"
        approximator_list.append(polyshap_3add_20)
    if "PolySHAP-3ADD-50%" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(0.5 * comb(n_players, 3))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_50 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_50.name = "PolySHAP-3ADD-50%"
        approximator_list.append(polyshap_3add_50)
    if "PolySHAP-3ADD-75%" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(0.75 * comb(n_players, 3))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_75 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_75.name = "PolySHAP-3ADD-75%"
        approximator_list.append(polyshap_3add_75)

    # define baselines
    if "PermutationSampling" in APPROXIMATORS:
        # Permutation Sampling
        permutation_sampling = PermutationSamplingSV(
            n=n_players, pairing_trick=PAIRING, random_state=RANDOM_STATE
        )
        permutation_sampling.name = "PermutationSampling"
        approximator_list.append(permutation_sampling)

    if "MSR" in APPROXIMATORS:
        # MSR = SHAPIQ order 1
        msr = UnbiasedKernelSHAP(
            n=n_players,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
        )
        msr.name = "MSR"
        approximator_list.append(msr)
    if "SVARM" in APPROXIMATORS:
        # SVARM
        svarm = SVARM(
            n=n_players,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
        )
        svarm.name = "SVARM"
        approximator_list.append(svarm)
    if "RegressionMSR" in APPROXIMATORS:
        # RegressionMSR
        regression_msr = RegressionMSR(
            n=n_players,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
        )
        regression_msr.name = "RegressionMSR"
        approximator_list.append(regression_msr)

    return approximator_list
