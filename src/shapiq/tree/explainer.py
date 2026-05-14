"""Implementation of the TreeExplainer class.

The :class:`~shapiq.tree.explainer.TreeSHAPIQ` uses the
:class:`~shapiq.tree.treeshapiq.TreeSHAPIQ` algorithm for computing any-order Interactions
for tree ensembles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from shapiq.explainer.base import Explainer
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

from .linear import LinearTreeSHAP
from .treeshapiq import TreeSHAPIQ, TreeSHAPIQIndices
from .validation import validate_tree_model

if TYPE_CHECKING:
    import numpy as np

    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import Model

    from .base import TreeModel

TREE_MODES = Literal["pathdependent", "interventional"]


class TreeExplainer(Explainer):
    """The TreeExplainer class for tree-based models.

    The explainer for tree-based models using the
    :class:`~shapiq.tree.treeshapiq.TreeSHAPIQ` algorithm. For details, refer to
    `Muschalik et al. (2024)` [Mus24]_.

    TreeSHAP-IQ is an algorithm for computing Shapley Interaction values for tree-based models.
    It is based on the Linear TreeSHAP algorithm by `Yu et al. (2022)` [Yu22]_, but extended to
    compute Shapley Interaction values up to a given order. TreeSHAP-IQ needs to visit each node
    only once and makes use of polynomial arithmetic to compute the Shapley Interaction values
    efficiently.

    The TreeExplainer can be used with a variety of tree-based models, including
    ``scikit-learn``, ``XGBoost``, ``LightGBM``, and ``CatBoost``. The explainer can handle both
    regression and classification models.

    References:
        .. [Yu22] Peng Yu, Chao Xu, Albert Bifet, Jesse Read. (2022). Linear Tree Shap. In: Proceedings of 36th Conference on Neural Information Processing Systems. https://openreview.net/forum?id=OzbkiUo24g
        .. [Mus24] Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, & Eyke Hüllermeier (2024). Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles. In: Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14388-14396. https://doi.org/10.1609/aaai.v38i13.29352

    """

    def __init__(
        self,
        model: dict | TreeModel | list[TreeModel] | Model,
        *,
        mode: TREE_MODES = "pathdependent",
        reference_dataset: np.ndarray | None = None,
        max_order: int = 2,
        min_order: int = 0,
        index: TreeSHAPIQIndices = "k-SII",
        class_index: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the TreeExplainer.

        Args:
            model: A tree-based model to explain.

            mode: The mode of the explainer, either ``"pathdependent"`` or ``"interventional"``.
            In ``"pathdependent"`` mode, the explainer computes path-dependent interaction values using the TreeSHAPIQ algorithm or the Linear TreeSHAP algorithm if the index is ``"SV"``.
            In ``"interventional"`` mode, the explainer computes interventional interaction values using the Interventional TreeExplainer algorithm.
            Defaults to ``"pathdependent"``.

            max_order: The maximum interaction order to be computed. An interaction order of ``1``
                corresponds to the Shapley value. Any value higher than ``1`` computes the Shapley
                interaction values up to that order. Defaults to ``2``.

            min_order: The minimum interaction order to keep in the returned
                :class:`~shapiq.interaction_values.InteractionValues`. Must satisfy
                ``0 <= min_order <= max_order``. When ``min_order == 0`` the empty interaction
                ``()`` is included with the baseline value. When ``min_order >= 1`` all
                interactions of order below ``min_order`` are filtered out of the result; the
                underlying algorithm still computes them internally when required by aggregated
                indices such as ``"k-SII"``. Defaults to ``0``.

            index: The type of interaction to be computed. It can be one of
                ``["k-SII", "SII", "STII", "FSII", "BII", "SV"]``. All indices apart from ``"BII"``
                will reduce to the ``"SV"`` (Shapley value) for order 1. Defaults to ``"k-SII"``.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            reference_dataset: A dataset to be used for reference in the explanation when using `mode=interventional`. Defaults to ``None``.

            **kwargs: Additional keyword arguments are ignored.

        """
        super().__init__(model, index=index, max_order=max_order)

        if min_order < 0 or min_order > self._max_order:
            msg = (
                f"min_order={min_order} must satisfy 0 <= min_order <= max_order "
                f"(max_order={self._max_order})."
            )
            raise ValueError(msg)

        # validate and parse model
        self._trees: list[TreeModel] = validate_tree_model(model, class_label=class_index)
        self._n_trees = len(self._trees)

        self._min_order: int = min_order
        self._class_label: int | None = class_index
        self.mode = mode
        self._reference_dataset: np.ndarray | None = reference_dataset

        # In ``"pathdependent"`` mode, build exactly one per-tree explainer list — either
        # ``LinearTreeSHAP`` (cheap, order-1 only) or ``TreeSHAPIQ`` (any order). The dispatch
        # decision is fixed at construction time so callers can mutate the chosen list (e.g.
        # ``_tree.thresholds`` rounding in tests) before calling :meth:`explain`. In
        # ``"interventional"`` mode no per-tree list is created — the
        # :class:`~shapiq.tree.interventional.explainer.InterventionalTreeExplainer` handles the
        # full ensemble in one shot, so a per-tree list would be meaningless.
        self._treeshapiq_explainers: list[TreeSHAPIQ] = []
        self._lineartreeshap_explainers: list[LinearTreeSHAP] = []
        self._interventional_explainer: InterventionalTreeExplainer | None = None

        if self.mode == "pathdependent":
            if self._can_use_lineartreeshap():
                self._lineartreeshap_explainers = [
                    LinearTreeSHAP(model=tree) for tree in self._trees
                ]
            else:
                # ``index`` (the local parameter) is already narrowed to ``TreeSHAPIQIndices``;
                # ``self.index`` is the broader ``ExplainerIndices`` and would not type-check.
                self._treeshapiq_explainers = [
                    TreeSHAPIQ(model=tree, max_order=self._max_order, index=index)
                    for tree in self._trees
                ]
        elif self.mode == "interventional":
            if self._reference_dataset is None:
                msg = (
                    "InterventionalTreeExplainer requires a reference_dataset; pass one to "
                    "TreeExplainer(..., mode='interventional', reference_dataset=...)."
                )
                raise ValueError(msg)
            self._interventional_explainer = InterventionalTreeExplainer(
                model=self._trees,
                data=self._reference_dataset,
                class_index=self._class_label,
                max_order=self._max_order,
                index=self.index,
            )

        # Baseline is the sum of the per-tree empty predictions and is identical regardless of
        # which algorithm runs explain — derive it from the trees directly so the attribute is
        # always populated, including in ``"interventional"`` mode where no per-tree list exists.
        self.baseline_value: float = float(sum(tree.empty_prediction for tree in self._trees))

    def _can_use_lineartreeshap(self) -> bool:
        """Whether the LinearTreeSHAP fast path can replace TreeSHAP-IQ for this configuration.

        LinearTreeSHAP is restricted to first-order Shapley values and needs at least two
        distinct features per tree (its Chebyshev base ``chebpts2`` requires ``npts >= 2``).
        Trivial trees (constant or single-feature) and higher-order interactions fall back to
        TreeSHAP-IQ, which carries dedicated trivial-tree fast paths.
        """
        return (
            self._max_order == 1
            and self.index in ("SV", "SII")
            and all(tree.n_features_in_tree >= 2 for tree in self._trees)
        )

    def _explain_function_lineartreeshap(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Compute first-order Shapley values for ``x`` by aggregating the per-tree LinearTreeSHAP results.

        Mirrors the per-tree aggregation done by ``_explain_function_treeshapiq``: each
        ``LinearTreeSHAP`` in ``self._lineartreeshap_explainers`` runs against ``x``, the
        resulting :class:`~shapiq.interaction_values.InteractionValues` are summed (which also
        sums ``baseline_value`` and the ``()`` entry), and ``min_order`` is finally enforced via
        :meth:`InteractionValues.get_n_order` when the user asked for a stricter minimum.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            The aggregated Shapley values for the instance.
        """
        if len(x.shape) != 1:
            msg = "explain expects a single instance, not a batch."
            raise TypeError(msg)

        interaction_values: list[InteractionValues] = [
            lts.explain_function(x) for lts in self._lineartreeshap_explainers
        ]
        final_explanation = interaction_values[0]
        for iv in interaction_values[1:]:
            final_explanation += iv

        if self._min_order > final_explanation.min_order:
            final_explanation = final_explanation.get_n_order(
                min_order=self._min_order,
                max_order=self._max_order,
            )

        return final_explanation

    def _explain_function_interventionaltreeshapiq(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Compute interaction values for ``x`` via the eagerly-built :class:`InterventionalTreeExplainer`.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            The interaction values for the instance.
        """
        if self._interventional_explainer is None:
            msg = "Interventional explainer is not initialized; mode must be 'interventional'."
            raise RuntimeError(msg)
        return self._interventional_explainer.explain_function(x)

    def _explain_function_treeshapiq(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Computes the Shapley Interaction values for a single instance.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            The interaction values for the instance.

        """
        if len(x.shape) != 1:
            msg = "explain expects a single instance, not a batch."
            raise TypeError(msg)
        # run treeshapiq for all trees
        interaction_values: list[InteractionValues] = []
        for explainer in self._treeshapiq_explainers:
            tree_explanation = explainer.explain(x)
            interaction_values.append(tree_explanation)

        # combine the explanations for all trees
        final_explanation = interaction_values[0]
        if len(interaction_values) > 1:
            for i in range(1, len(interaction_values)):
                final_explanation += interaction_values[i]

        if self._min_order == 0 and final_explanation.min_order == 1:
            final_explanation.min_order = 0
            # Add the baseline value to the empty prediction
            # might break for some edge cases
            final_explanation.interactions[()] = float(final_explanation.baseline_value)

        if self._min_order > final_explanation.min_order:
            final_explanation = final_explanation.get_n_order(
                min_order=self._min_order,
                max_order=self._max_order,
            )

        return final_explanation

    def explain_function(  # type: ignore[override]
        self,
        x: np.ndarray,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,
    ) -> InteractionValues:
        """Computes the interaction index for a single instance.

        The method used for computing the explanation depends on the specified mode and the
        parameters of the explainer.

        Args:
            x: The instance to explain as a 1-dimensional array.
            *args: Additional positional arguments are ignored.
            **kwargs: Additional keyword arguments forwarded to the per-mode explain function.

        Returns:
            The computed interaction index for the instance.
        """
        if self.mode == "pathdependent":
            # Dispatch on whichever per-tree list __init__ chose to populate.
            if self._lineartreeshap_explainers:
                return self._explain_function_lineartreeshap(x, **kwargs)
            return self._explain_function_treeshapiq(x, **kwargs)
        return self._explain_function_interventionaltreeshapiq(x, **kwargs)
