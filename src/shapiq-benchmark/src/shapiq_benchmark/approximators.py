from __future__ import annotations


from lightgbm import LGBMRegressor
import numpy as np
from shapiq.approximator import SVARMIQ, SHAPIQ, KernelSHAPIQ, ProxySPEX, ProxySHAP


def get_approximators(
    APPROXIMATORS, NPLAYERS, RANDOMSTATE, PAIRING, INDEX, MAXORDER, n_estimators=None
):
    approximator_list = []
    sampling_weights = np.ones(NPLAYERS + 1)
    if "SVARM" in APPROXIMATORS or "SVARMIQ" in APPROXIMATORS:
        svarmiq = SVARMIQ(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
        )
        svarmiq.name = "SVARMIQ"
        approximator_list.append(svarmiq)
    if "SHAPIQ" in APPROXIMATORS:
        shapiq = SHAPIQ(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
        )
        shapiq.name = "SHAPIQ"
        approximator_list.append(shapiq)
    if "ProxySPEX" in APPROXIMATORS:
        proxy_spex = ProxySPEX(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            index=INDEX,
            max_order=MAXORDER,
        )
        proxy_spex.name = "ProxySPEX"
        approximator_list.append(proxy_spex)
    if "KernelSHAPIQ" in APPROXIMATORS:
        kernel_shapiq = KernelSHAPIQ(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
        )
        kernel_shapiq.name = "KernelSHAPIQ"
        approximator_list.append(kernel_shapiq)
    if "ProxySHAP" in APPROXIMATORS:
        proxyshap = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
        )
        proxyshap.name = "ProxySHAP"
        approximator_list.append(proxyshap)
    if "ProxySHAP-NoAdjustment" in APPROXIMATORS:
        proxyshap_no_adjust = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            adjustment="none",
            sampling_weights=sampling_weights,
        )
        proxyshap_no_adjust.name = "ProxySHAP-NoAdjust"
        approximator_list.append(proxyshap_no_adjust)
    if "ConsistentTree" in APPROXIMATORS:
        from consitent_tree import ConsistentTree

        ct_kwargs = {}
        if n_estimators is not None:
            ct_kwargs["n_estimators"] = n_estimators
        consistent_tree = ConsistentTree(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            value_weighting="shapley",
            marginal_weighting="shapley",
            lambda_value=1,
            lambda_marginal=1,
            hessian_mode="rowsum",
            mode="mixed",
            **ct_kwargs,
        )
        consistent_tree.name = "ConsistentTree"
        approximator_list.append(consistent_tree)
    if "ConsistentTree-Value" in APPROXIMATORS:
        from consitent_tree import ConsistentTree

        ct_kwargs = {}
        if n_estimators is not None:
            ct_kwargs["n_estimators"] = n_estimators
        consistent_tree = ConsistentTree(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            value_weighting="shapley",
            marginal_weighting="shapley",
            lambda_value=1,
            lambda_marginal=1,
            hessian_mode="rowsum",
            mode="value",
            **ct_kwargs,
        )
        consistent_tree.name = "ConsistentTree-Value"
        approximator_list.append(consistent_tree)
    if "ConsistentTree-Value-KernelSHAP" in APPROXIMATORS:
        from consitent_tree import ConsistentTree

        ct_kwargs = {}
        if n_estimators is not None:
            ct_kwargs["n_estimators"] = n_estimators
        consistent_tree = ConsistentTree(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            value_weighting="kernelshap",
            marginal_weighting="shapley",
            lambda_value=1,
            lambda_marginal=1,
            hessian_mode="rowsum",
            mode="value",
            **ct_kwargs,
        )
        consistent_tree.name = "ConsistentTree-Value-KernelSHAP"
        approximator_list.append(consistent_tree)
    if "ConsistentTree-Marginal" in APPROXIMATORS:
        from consitent_tree import ConsistentTree

        ct_kwargs = {}
        if n_estimators is not None:
            ct_kwargs["n_estimators"] = n_estimators
        consistent_tree = ConsistentTree(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            value_weighting="shapley",
            marginal_weighting="shapley",
            lambda_value=1,
            lambda_marginal=1,
            hessian_mode="rowsum",
            mode="marginal",
            **ct_kwargs,
        )
        consistent_tree.name = "ConsistentTree-Marginal"
        approximator_list.append(consistent_tree)
    if "ConsistentTree-Simple-InverseBinom" in APPROXIMATORS:
        from consitent_tree import ConsistentTreeSimpler

        ct_kwargs = {}
        if n_estimators is not None:
            ct_kwargs["n_estimators"] = n_estimators
        consistent_tree = ConsistentTreeSimpler(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            weighting="inverse_binom",
            alpha=0.5,
            **ct_kwargs,
        )
        consistent_tree.name = "ConsistentTree-Simple-InverseBinom"
        approximator_list.append(consistent_tree)
    if "ConsistentTree-Simple-KernelSHAP" in APPROXIMATORS:
        from consitent_tree import ConsistentTreeSimpler

        ct_kwargs = {}
        if n_estimators is not None:
            ct_kwargs["n_estimators"] = n_estimators
        consistent_tree = ConsistentTreeSimpler(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            weighting="kernelshap",
            alpha=0.5,
            **ct_kwargs,
        )
        consistent_tree.name = "ConsistentTree-Simple-KernelSHAP"
        approximator_list.append(consistent_tree)
    if "ConsistentTree-Two-Stage" in APPROXIMATORS:
        from consitent_tree import ConsistentTreeTwoStage

        ct_kwargs = {}
        if n_estimators is not None:
            ct_kwargs["n_estimators"] = n_estimators
        consistent_tree = ConsistentTreeTwoStage(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            objective="mixed",
            value_weighting="kernelshap",
            marginal_weighting="shapley",
            lambda_value=1,
            lambda_marginal=1,
            lambda_shapley=0,
            **ct_kwargs,
        )
        consistent_tree.name = "ConsistentTree-Two-Stage"
        approximator_list.append(consistent_tree)
    return approximator_list
