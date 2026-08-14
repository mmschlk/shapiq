"""Implementation of the TreeExplainer class.

The :class:`~shapiq.tree.explainer.TreeSHAPIQ` uses the
:class:`~shapiq.tree.treeshapiq.TreeSHAPIQ` algorithm for computing any-order Interactions
for tree ensembles.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from shap.utils import safe_isinstance

from shapiq.explainer.base import Explainer
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

from .base import TreeModel
from .linear import LinearTreeSHAP
from .treeshapiq import TreeSHAPIQ
from .validation import validate_tree_model

if TYPE_CHECKING:
    from shapiq.typing import Model

TREE_MODES = Literal["pathdependent", "interventional"]
TreeExplainerIndices = Literal["SV", "SII", "k-SII", "BV", "BII"]


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

    On large datasets the explainer relays on the Woodelf and WoodelfHD algorithms. For details, refer to
    `Nadel and Wettenstein (2026)` [Nad26]_ and `Wettenstein et al. (2026)` [Wet26]_


    References:
        .. [Yu22] Peng Yu, Chao Xu, Albert Bifet, Jesse Read. (2022). Linear Tree Shap. In: Proceedings of 36th Conference on Neural Information Processing Systems. https://openreview.net/forum?id=OzbkiUo24g
        .. [Mus24] Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, & Eyke Hüllermeier (2024). Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles. In: Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14388-14396. https://doi.org/10.1609/aaai.v38i13.29352
        ... [Nad26] Nadel, A., & Wettenstein, R. (2026). From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm. Proceedings of the AAAI Conference on Artificial Intelligence, 40(29), 24476–24485. https://doi.org/10.1609/aaai.v40i29.39630
        .... [Wet26] Ron Wettenstein, Alexander Nadel, Udi Boker. (2026). WOODELF-HD: Efficient Background SHAP for High-Depth Decision Trees. arXiv preprint arXiv:2604.10569. https://arxiv.org/abs/2604.10569

    """

    def __init__(
        self,
        model: dict | TreeModel | list[TreeModel] | Model,
        *,
        mode: TREE_MODES = "pathdependent",
        reference_dataset: np.ndarray | None = None,
        max_order: int = 1,
        min_order: int = 0,
        index: TreeExplainerIndices = "SV",
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

            max_order: The maximum order of interactions to be computed. Set to ``1`` for no
                interactions (i.e, for Shapley values ``"SV"`` or Banzhaf values ``"BV"``). Any
                value higher than ``1`` computes interaction values up to that order. Defaults to
                ``1``.

            min_order: The minimum interaction order to keep in the returned
                :class:`~shapiq.interaction_values.InteractionValues`. Must satisfy
                ``0 <= min_order <= max_order``. When ``min_order == 0`` the empty interaction
                ``()`` is included with the baseline value. When ``min_order >= 1`` all
                interactions of order below ``min_order`` are filtered out of the result; the
                underlying algorithm still computes them internally when required by aggregated
                indices such as ``"k-SII"``. Defaults to ``0``.

            index: The type of interaction to be computed. It can be one of
                ``["k-SII", "SII", "STII", "FSII", "BII", "SV"]``. All indices apart from ``"BII"``
                will reduce to the ``"SV"`` (Shapley value) for order 1. Defaults to ``"SV"``.

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

        # Set when treelite fails to load its native library inside ``_run_woodelf``; keeps every
        # later explain call on the shapiq kernels instead of re-attempting a dlopen that cannot
        # succeed within this process.
        self._woodelf_unavailable: bool = False

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
        self.explainers_initialized = False

        # Baseline is the sum of the per-tree empty predictions and is identical regardless of
        # which algorithm runs explain — derive it from the trees directly so the attribute is
        # always populated, including in ``"interventional"`` mode where no per-tree list exists.
        self.baseline_value: float = float(sum(tree.empty_prediction for tree in self._trees))

    def _init_explainers(self, index: Literal["SV", "SII", "k-SII"]):
        self.explainers_initialized = True
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

    def _should_use_woodelf(self, number_of_explained_instances):
        """The function decide when to use Woodelf and when to use shapiq implementation for Shapley values computation.
        The cut-offs are n*m >= 100 for interventional and n >= 100 for path dependent where n is the number of explained instances and
        m is the size of the reference dataset. They are based on experiments summarized in the htmls below:

        Path Dependent experiment:
        https://ron-wettenstein.github.io/TreeBranchMarks/benchmarks/reports/woodelf_vs_shapiq_path_dependent_experiment.html

        Interventional experiment:
        https://ron-wettenstein.github.io/TreeBranchMarks/benchmarks/reports/woodelf_vs_shapiq_experiment.html

        This function should change when new capabilities are developed in Woodelf.
        """
        if self._woodelf_unavailable:
            return False

        if self.index not in ("SV", "BV", "SII", "BII"):
            return False

        # Woodelf currently does not support cat boost models
        cat_boost_classes = [
            "catboost.core.CatBoostRegressor",
            "catboost.core.CatBoostClassifier",
            "catboost.core.CatBoost",
        ]
        if any(safe_isinstance(self.model, catboost_cls) for catboost_cls in cat_boost_classes):
            return False

        # Woodelf needs the original model as an input
        if isinstance(self.model, list) or isinstance(self.model, TreeModel):
            return False

        if self.mode == "interventional":
            if len(self._reference_dataset) * number_of_explained_instances >= 100:
                return True
        elif self.mode == "pathdependent":
            if self.max_order > 1 or number_of_explained_instances >= 100:
                return True
            if self.index in ("BV", "BII"):
                return True
        return False

    def _run_woodelf(self, X: np.ndarray):
        """Compute Shapley or Banzhaf values using the Woodelf package.
        We use the hybrid_woodelf function.

        Return the values in Woodelf format or None if this configuration is not supported in Woodelf (e.g. unsupported index or too deep tree)
        """
        try:
            import pandas as pd
            from woodelf.core.cube_metric import (
                BanzhafValues,
                GeneralBanzhafInteractionValues,
                GeneralShapleyInteractionValues,
                ShapleyValues,
            )
            from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
            from woodelf.woodelf_sparse import hybrid_woodelf
        except ImportError:
            raise ImportError(
                "For efficient computation of decision trees woodelf and treelite needs to be installed.\n"
                + "You can install all the needed package for decision tree explainability by installing shapiq [trees] extra. Run: \n"
                + ">> pip install shapiq[trees]\n"
                + "Note: the woodelf package is published on PyPI as 'woodelf_explainer'; "
                + "'pip install woodelf' installs an unrelated package."
            )

        consumer_dataset = pd.DataFrame(X)
        background_dataset = None
        if self._reference_dataset is not None and self.mode == "interventional":
            background_dataset = pd.DataFrame(self._reference_dataset)

        if self._index == "SV":
            metric = ShapleyValues()
        elif self._index == "BV":
            metric = BanzhafValues()
        elif self._index == "SII":
            metric = GeneralShapleyInteractionValues(max(self._min_order, 1), self._max_order)
        elif self._index == "BII":
            metric = GeneralBanzhafInteractionValues(max(self._min_order, 1), self._max_order)
        else:
            return None

        class_index = self._class_label if self._class_label is not None else 1
        try:
            loaded_model = load_decision_tree_ensemble_model(
                self.model, range(X.shape[1]), class_index=class_index
            )
        except OSError as error:
            # Woodelf imports treelite lazily during model parsing, and treelite dlopens its
            # native library at import time -- so a load failure surfaces here as an OSError at
            # explain time, not as an ImportError above. Known case: treelite's macOS wheels
            # link ``@rpath/libomp.dylib`` but neither bundle libomp nor bake in a usable rpath.
            self._woodelf_unavailable = True
            warnings.warn(
                "Woodelf is installed, but treelite failed to load its native library:\n"
                f"    {error}\n"
                "Falling back to shapiq's own implementation: results are identical, but "
                "computation is slower on large datasets.\n"
                "On macOS, treelite needs the OpenMP runtime, which its wheels do not bundle. "
                "To enable the Woodelf fast path, install libomp and make it visible to the "
                "dynamic linker before starting Python:\n"
                "    brew install libomp\n"
                '    export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix libomp)/lib:'
                '$DYLD_FALLBACK_LIBRARY_PATH"',
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        # TODO when woodelf support path dependent SHAP on high depth trees remove this if
        if self.mode == "pathdependent" and self.max_order >= 3 and loaded_model.max_depth > 16:
            return None

        woodelf_result = hybrid_woodelf(
            model=loaded_model,
            consumer_data=consumer_dataset,
            background_data=background_dataset,
            metric=metric,
            model_was_loaded=True,
        )
        if self._index in ("SV", "BV"):
            return {(k,): v for k, v in woodelf_result.items()}
        return woodelf_result

    def _cast_shapiq_results_to_woodelf_format(
        self,
        shapiq_results: list[InteractionValues],
    ) -> dict[tuple, np.ndarray]:
        """Transpose one shapiq output format, ``InteractionValues`` per row, into ``_run_woodelf``'s output format, ``{interaction_tuple: ndarray(n_instances,)}``,
        keeping orders ``max(min_order, 1) .. max_order`` (which also drops the ``()`` baseline).
        """
        lowest = max(self._min_order, 1)

        # Shapiq omits exactly-zero terms per row and woodelf include them (it emits only terms that are zero in all rows)
        all_subsets = {
            subset
            for result in shapiq_results
            for subset in result.interactions
            if lowest <= len(subset) <= self._max_order
        }

        woodelf_format = {subset: np.zeros(len(shapiq_results)) for subset in all_subsets}

        # A subset a row does not report stays 0.0 -- that is what its absence means.
        for row, result in enumerate(shapiq_results):
            for subset, value in result.interactions.items():
                if subset in woodelf_format:
                    woodelf_format[subset][row] = value

        return woodelf_format

    def _cast_woodelf_result_to_shapiq_format(
        self, woodelf_result: dict[tuple, np.ndarray], n_players
    ) -> InteractionValues:
        interaction_values = {
            subset: values[0] for subset, values in woodelf_result.items() if values[0] != 0
        }
        if self._min_order == 0:
            interaction_values[tuple()] = self.baseline_value

        return InteractionValues(
            values=interaction_values,
            index=self._index,
            max_order=self._max_order,
            n_players=n_players,
            min_order=self._min_order,
            baseline_value=self.baseline_value,
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
        if self._should_use_woodelf(number_of_explained_instances=1):
            woodelf_explanation = self._run_woodelf(np.array([x]))
            if woodelf_explanation is not None:
                return self._cast_woodelf_result_to_shapiq_format(
                    woodelf_explanation, n_players=int(x.shape[0])
                )

        if not self.explainers_initialized:
            self._init_explainers(self.index)
        if self.mode == "pathdependent":
            # Dispatch on whichever per-tree list __init__ chose to populate.
            if self._lineartreeshap_explainers:
                return self._explain_function_lineartreeshap(x, **kwargs)
            return self._explain_function_treeshapiq(x, **kwargs)
        return self._explain_function_interventionaltreeshapiq(x, **kwargs)

    def explain_X(
        self,
        X: np.ndarray,
        *,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[tuple, np.ndarray]:
        """Explaining many instances and once, using Woodelf on larger datasets and shapiq on smaller onces"""
        if self._should_use_woodelf(len(X)):
            woodelf_result = self._run_woodelf(X)
            if woodelf_result is not None:
                return woodelf_result

        shapiq_results = super().explain_X(
            X, n_jobs=n_jobs, random_state=random_state, verbose=verbose, **kwargs
        )
        return self._cast_shapiq_results_to_woodelf_format(shapiq_results)
