from __future__ import annotations

import numpy as np
from scipy.special import binom, comb

from shapiq import PermutationSamplingSV, UnbiasedKernelSHAP, SVARM, KernelSHAP
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

    frontier_generator = ExplanationFrontierGenerator(N=set(range(n_players)), random_state = RANDOM_STATE)

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

    if "OldLeverageSHAP" in APPROXIMATORS:
        # LeverageSHAP
        leverage_shap = KernelSHAP(
            n_players,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        leverage_shap.name = "OldLeverageSHAP"
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
    if "PolySHAP-5ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        explanation_frontier = frontier_generator.generate_kadd(max_order=5)
        polyshap_5add = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_5add.name = "PolySHAP-5ADD"
        approximator_list.append(polyshap_5add)

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
    if "PolySHAP-5ADD-10%" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + comb(n_players, 3) + comb(n_players,4) + int(0.1 * comb(n_players, 5))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_5add_10 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_5add_10.name = "PolySHAP-5ADD-10%"
        approximator_list.append(polyshap_5add_10)
    if "PolySHAP-3ADD-dlog(d)/2" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(n_players/2*np.log(comb(n_players, 3)))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_dloghalf = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_dloghalf.name = "PolySHAP-3ADD-dlog(d)/2"
        approximator_list.append(polyshap_3add_dloghalf)
    if "PolySHAP-3ADD-dlog(d)" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(n_players*np.log(comb(n_players, 3)))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_dlog = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_dlog.name = "PolySHAP-3ADD-dlog(d)"
        approximator_list.append(polyshap_3add_dlog)
    if "PolySHAP-3ADD-2dlog(d)" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(2*n_players*np.log(comb(n_players, 3)))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_2dlog = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_2dlog.name = "PolySHAP-3ADD-2dlog(d)"
        approximator_list.append(polyshap_3add_2dlog)
    if "PolySHAP-3ADD-dlog(d)" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(3*n_players*np.log(comb(n_players, 3)))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_3dlog = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_3dlog.name = "PolySHAP-3ADD-3dlog(d)"
        approximator_list.append(polyshap_3add_3dlog)
    if "PolySHAP-3ADD-dlog(d)sqrt(d)" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(np.sqrt(n_players)*n_players*np.log(comb(n_players, 3)))
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_dlogsqrt = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_dlogsqrt.name = "PolySHAP-3ADD-dlog(d)sqrt(d)"
        approximator_list.append(polyshap_3add_dlogsqrt)
    if "PolySHAP-3ADD-5d" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(5*n_players)
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_5d = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_5d.name = "PolySHAP-3ADD-5d"
        approximator_list.append(polyshap_3add_5d)
    if "PolySHAP-3ADD-20d" in APPROXIMATORS:
        n_coefficients = (
            1 + n_players + comb(n_players, 2) + int(20*n_players)
        )
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_20d = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_20d.name = "PolySHAP-3ADD-20d"
        approximator_list.append(polyshap_3add_20d)
    if "PolySHAP-3ADD-3000" in APPROXIMATORS:
        binom3 = (
             1 + n_players + comb(n_players, 2) + comb(n_players,3)
        )
        n_coefficients = min(3000,binom3)
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_3000 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_3000.name = "PolySHAP-3ADD-3000"
        approximator_list.append(polyshap_3add_3000)
    if "PolySHAP-3ADD-4000" in APPROXIMATORS:
        binom3 = (
             1 + n_players + comb(n_players, 2) + comb(n_players,3)
        )
        n_coefficients = min(4000,binom3)
        explanation_frontier = frontier_generator.generate_partial(
            n_explanation_terms=n_coefficients
        )
        # ShapleyGAX with leverage weights for order 1
        polyshap_3add_4000 = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add_4000.name = "PolySHAP-3ADD-4000"
        approximator_list.append(polyshap_3add_4000)
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
            sampling_weights=sampling_weights_leverage_1,
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
            sampling_weights=sampling_weights_leverage_1,
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
            sampling_weights=sampling_weights_leverage_1,
        )
        regression_msr.name = "RegressionMSR"
        approximator_list.append(regression_msr)

    return approximator_list
