"""Implementation of the TreeExplainer class.

The :class:`~shapiq.tree.explainer.TreeSHAPIQ` uses the
:class:`~shapiq.tree.treeshapiq.TreeSHAPIQ` algorithm for computing any-order Interactions
for tree ensembles.
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.explainer.base import Explainer
from shapiq.interaction_values import InteractionValues, InteractionValuesBatch
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq.utils.modules import safe_isinstance

from .base import TreeModel
from .linear import LinearTreeSHAP
from .treeshapiq import TreeSHAPIQ
from .validation import validate_tree_model

if TYPE_CHECKING:
    from shapiq.typing import Model

TREE_MODES = Literal["pathdependent", "interventional"]
TREE_BACKENDS = Literal["auto", "woodelf", "shapiq"]
TreeExplainerIndices = Literal["SV", "SII", "k-SII", "BV", "BII"]

_WOODELF_INSTALL_HINT = "Install it with: pip install shapiq[tree]"
_WOODELF_REQUIRED = f"requires the optional 'woodelf-explainer' package. {_WOODELF_INSTALL_HINT}"


class WoodelfNotAvailableWarning(UserWarning):
    """The explanation would be computed faster with the optional woodelf dependency.

    Emitted when an input crosses :class:`TreeExplainer`'s Woodelf cut-offs but the optional
    ``woodelf-explainer`` package is not installed, so the (slower) shapiq implementation
    computes the explanation instead. Filter it with
    ``warnings.filterwarnings("ignore", category=WoodelfNotAvailableWarning)`` or silence it
    for good by installing ``shapiq[tree]``.
    """


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

    On large datasets the explainer relies on the Woodelf and WOODELF-HD algorithms. For
    details, refer to `Nadel and Wettenstein (2026)` [Nad26]_ and
    `Wettenstein et al. (2026)` [Wet26]_.

    References:
        .. [Yu22] Peng Yu, Chao Xu, Albert Bifet, Jesse Read. (2022). Linear Tree Shap. In: Proceedings of 36th Conference on Neural Information Processing Systems. https://openreview.net/forum?id=OzbkiUo24g
        .. [Mus24] Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, & Eyke Hüllermeier (2024). Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles. In: Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14388-14396. https://doi.org/10.1609/aaai.v38i13.29352
        .. [Nad26] Nadel, A., & Wettenstein, R. (2026). From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm. Proceedings of the AAAI Conference on Artificial Intelligence, 40(29), 24476-24485. https://doi.org/10.1609/aaai.v40i29.39630
        .. [Wet26] Ron Wettenstein, Alexander Nadel, Udi Boker. (2026). WOODELF-HD: Efficient Background SHAP for High-Depth Decision Trees. arXiv preprint arXiv:2604.10569. https://arxiv.org/abs/2604.10569

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
        backend: TREE_BACKENDS = "auto",
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
                ``["SV", "SII", "k-SII", "BV", "BII"]``. Both ``"SII"`` and ``"k-SII"``
                reduce to the ``"SV"`` (Shapley value) for order 1. Defaults to ``"SV"``.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            reference_dataset: A dataset to be used for reference in the explanation. Required
                when ``mode="interventional"``. Defaults to ``None``.

            backend: Which implementation computes the explanations. With ``"auto"`` the
                explainer picks per input: Woodelf on larger inputs (falling back to shapiq
                with a :class:`WoodelfNotAvailableWarning` if the optional woodelf dependency
                is missing) and shapiq otherwise. ``"woodelf"`` forces Woodelf and raises if it
                is not installed or cannot handle the configuration; ``"shapiq"`` forces the
                shapiq implementation. Defaults to ``"auto"``.

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
        if mode == "interventional" and reference_dataset is None:
            msg = (
                "mode='interventional' requires a reference_dataset; pass one to "
                "TreeExplainer(..., mode='interventional', reference_dataset=...)."
            )
            raise ValueError(msg)
        self._mode = mode
        self._reference_dataset: np.ndarray | None = reference_dataset

        if backend not in ("auto", "woodelf", "shapiq"):
            msg = f"backend='{backend}' must be one of 'auto', 'woodelf', or 'shapiq'."
            raise ValueError(msg)
        self.backend: TREE_BACKENDS = backend
        if backend == "woodelf":
            # forced means forced: fail fast instead of silently falling back later.
            if importlib.util.find_spec("woodelf") is None:
                msg = f"backend='woodelf' {_WOODELF_REQUIRED}"
                raise ImportError(msg)
            reason = self._woodelf_unsupported_reason()
            if reason is not None:
                msg = f"backend='woodelf' cannot be used: {reason}."
                raise ValueError(msg)
        elif mode == "pathdependent" and index in ("BV", "BII"):
            # only Woodelf computes path-dependent Banzhaf indices; fail fast when it is
            # excluded by choice or missing from the environment.
            if backend == "shapiq":
                msg = (
                    f"index='{index}' with mode='pathdependent' is only computed by the woodelf "
                    "backend; use backend='auto' or backend='woodelf'."
                )
                raise ValueError(msg)
            if importlib.util.find_spec("woodelf") is None:
                msg = f"index='{index}' with mode='pathdependent' {_WOODELF_REQUIRED}"
                raise ImportError(msg)

        self._treeshapiq_explainers: list[TreeSHAPIQ] = []
        self._lineartreeshap_explainers: list[LinearTreeSHAP] = []
        self._interventional_explainer: InterventionalTreeExplainer | None = None
        self._explainers_initialized = False

        self._baseline_value: float | None = None

    @property
    def baseline_value(self) -> float:
        """The empty prediction of the explained model, matching the explanation mode.

        Computed lazily on first access and cached. In ``"pathdependent"`` mode this is the sum
        of the per-tree (coverage-weighted) empty predictions; in ``"interventional"`` mode it is
        the mean ensemble prediction over the reference dataset.
        """
        if self._baseline_value is None:
            if self.mode == "interventional":
                if self._reference_dataset is None:
                    msg = (
                        "The interventional baseline value requires a reference_dataset; pass "
                        "one to TreeExplainer(..., mode='interventional', reference_dataset=...)."
                    )
                    raise ValueError(msg)
                self._baseline_value = InterventionalTreeExplainer.compute_empty_prediction(
                    self._trees, self._reference_dataset
                )
            else:
                self._baseline_value = float(sum(tree.empty_prediction for tree in self._trees))
        return self._baseline_value

    @property
    def mode(self) -> Literal["interventional", "pathdependent"]:
        """The mode of the explainer."""
        return self._mode

    def _init_explainers(self) -> None:
        """Build the shapiq explainers for the configured mode.

        Runs lazily on the first explanation that is not routed to Woodelf, so the (potentially
        expensive) shapiq explainers are never built when Woodelf handles the computation.
        """
        self._explainers_initialized = True
        if self.mode == "pathdependent":
            if self._can_use_lineartreeshap():
                self._lineartreeshap_explainers = [
                    LinearTreeSHAP(model=tree) for tree in self._trees
                ]
            else:
                index = self.index
                if index not in ("SV", "SII", "k-SII"):
                    msg = (
                        f"index='{index}' with mode='pathdependent' is only supported via the "
                        "optional woodelf dependency (`pip install shapiq[tree]`), which is "
                        "unavailable or does not support this configuration."
                    )
                    raise ValueError(msg)
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

    def _should_use_woodelf(self, number_of_explained_instances: int) -> bool:
        """Decide whether Woodelf or the shapiq implementation computes the explanation.

        The cut-offs are ``n * m >= 100`` for interventional and ``n >= 100`` (or
        ``max_order > 1``, or a Banzhaf index) for path-dependent mode, where ``n`` is the number
        of explained instances and ``m`` is the size of the reference dataset. They are based on
        the experiments summarized in the reports below:

        Path-dependent experiment:
        https://ron-wettenstein.github.io/TreeBranchMarks/benchmarks/reports/woodelf_vs_shapiq_path_dependent_experiment.html

        Interventional experiment:
        https://ron-wettenstein.github.io/TreeBranchMarks/benchmarks/reports/woodelf_vs_shapiq_experiment.html

        This function should change when new capabilities are developed in Woodelf.

        Args:
            number_of_explained_instances: How many instances are about to be explained.

        Returns:
            ``True`` if Woodelf should compute the explanation, ``False`` for shapiq.
        """
        if self.backend == "shapiq":
            return False
        if self.backend == "woodelf":
            return True

        if self._woodelf_unsupported_reason() is not None:
            return False

        if self.mode == "interventional":
            if (
                self._reference_dataset is not None
                and len(self._reference_dataset) * number_of_explained_instances >= 100
            ):
                return self._woodelf_available()
        elif self.mode == "pathdependent":
            if self.max_order > 1 or number_of_explained_instances >= 100:
                return self._woodelf_available()
            if self.index in ("BV", "BII"):
                return self._woodelf_available()
        return False

    def _woodelf_unsupported_reason(self) -> str | None:
        """The reason why Woodelf cannot serve this explainer's configuration, if any.

        Returns:
            A human-readable reason, or ``None`` when Woodelf supports the configuration.
        """
        if self.index not in ("SV", "BV", "SII", "k-SII", "BII"):
            return f"index='{self.index}' is not supported by Woodelf"

        cat_boost_classes = [
            "catboost.core.CatBoostRegressor",
            "catboost.core.CatBoostClassifier",
            "catboost.core.CatBoost",
        ]
        if any(safe_isinstance(self.model, catboost_cls) for catboost_cls in cat_boost_classes):
            return "Woodelf does not support CatBoost models"

        if isinstance(self.model, list | TreeModel):
            return "Woodelf requires the original model object, not already-parsed trees"

        return None

    @staticmethod
    def _woodelf_available() -> bool:
        """Return whether the optional woodelf dependency is installed.

        Only called once the cut-offs decided for Woodelf, so a missing package warns the user
        that the computation falls back to the (slower) shapiq implementation.
        """
        if importlib.util.find_spec("woodelf") is not None:
            return True
        warnings.warn(
            "This explanation would be computed substantially faster with the optional "
            f"'woodelf-explainer' package. {_WOODELF_INSTALL_HINT}",
            category=WoodelfNotAvailableWarning,
            stacklevel=2,
        )
        return False

    def _run_woodelf(self, X: np.ndarray) -> dict[tuple, np.ndarray] | None:
        """Compute Shapley or Banzhaf (interaction) values with Woodelf's ``hybrid_woodelf``.

        Args:
            X: The instances to explain as a 2-dimensional array of shape
                ``(n_instances, n_features)``.

        Returns:
            The values in the Woodelf format ``{interaction_tuple: ndarray of shape
            (n_instances,)}``, or ``None`` if the configuration is not supported by Woodelf
            (e.g. an unsupported index or a too deep tree).

        Raises:
            RuntimeError: If woodelf is installed but its treelite backend cannot load its
                native library (e.g. a missing OpenMP runtime on macOS).
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
        except ImportError as error:
            msg = f"The Woodelf fast path {_WOODELF_REQUIRED}"
            raise ImportError(msg) from error

        consumer_dataset = pd.DataFrame(X)
        background_dataset = None
        if self._reference_dataset is not None and self.mode == "interventional":
            background_dataset = pd.DataFrame(self._reference_dataset)

        if self._index == "SV":
            metric = ShapleyValues()
        elif self._index == "BV":
            metric = BanzhafValues()
        elif self._index in ("SII", "k-SII"):
            # k-SII is a pure aggregation of SII, Woodelf computes the SII base values and
            # ``_aggregate_batched_sii_to_ksii`` makes them k-SII
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
            # treelite imports lazily inside woodelf's model parsing, so a macOS wheel that
            # cannot load libomp surfaces here as an OSError.
            msg = (
                "woodelf is installed but its treelite backend failed to load. On macOS this "
                "usually means the OpenMP runtime is missing: install it with "
                "`brew install libomp` and set "
                "`DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix libomp)/lib`. "
                "See https://github.com/dmlc/treelite/issues/678 for details."
            )
            raise RuntimeError(msg) from error

        # woodelf cannot compute path-dependent SHAP of order >= 3 on trees deeper than 16
        # yet; remove this fallback once it can.
        if self.mode == "pathdependent" and self.max_order >= 3 and loaded_model.max_depth > 16:
            if self.backend == "woodelf":
                msg = (
                    "backend='woodelf' cannot compute path-dependent interactions of "
                    f"order >= 3 on trees deeper than 16 (tree depth: "
                    f"{loaded_model.max_depth}); use backend='auto' or backend='shapiq'."
                )
                raise ValueError(msg)
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
        if self._index == "k-SII":
            return self._aggregate_batched_sii_to_ksii(woodelf_result)
        return woodelf_result

    def _aggregate_batched_sii_to_ksii(
        self, sii_result: dict[tuple, np.ndarray]
    ) -> dict[tuple, np.ndarray]:
        """Aggregate a batched Woodelf SII result into k-SII values.

        k-SII is a linear aggregation of SII (the same one the shapiq explainers apply through
        :class:`~shapiq.interaction_values.InteractionValues`), so it transfers unchanged onto
        the per-row value arrays of the Woodelf format.

        Args:
            sii_result: Woodelf-format SII values ``{interaction_tuple: ndarray(n_instances,)}``.

        Returns:
            The k-SII values in the same format, restricted to orders
            ``max(min_order, 1) .. max_order``.
        """
        from shapiq.game_theory.aggregation import aggregate_base_attributions

        # min_order=1 describes the aggregation base, not the requested filter: the k-SII value
        # of an interaction only depends on the SII values of its supersets, which Woodelf all
        # computed, so the kept orders below are exact even when ``self._min_order > 1``.
        aggregated, _, _ = aggregate_base_attributions(
            interactions=sii_result,
            index="SII",
            order=self._max_order,
            min_order=1,
            baseline_value=self.baseline_value,
        )
        lowest = max(self._min_order, 1)
        # the isinstance check narrows out the scalar ``()`` baseline entry, which the order
        # filter drops anyway
        return {
            subset: values
            for subset, values in aggregated.items()
            if isinstance(values, np.ndarray) and lowest <= len(subset) <= self._max_order
        }

    def _cast_shapiq_results_to_woodelf_format(
        self,
        shapiq_results: list[InteractionValues],
    ) -> dict[tuple, np.ndarray]:
        """Transpose a per-row list of ``InteractionValues`` into the Woodelf output format.

        Keeps orders ``max(min_order, 1) .. max_order`` (which also drops the ``()`` baseline).

        Args:
            shapiq_results: One :class:`InteractionValues` object per explained instance.

        Returns:
            The values in the Woodelf format ``{interaction_tuple: ndarray of shape
            (n_instances,)}``.
        """
        lowest = max(self._min_order, 1)

        # shapiq omits exactly-zero terms per row and woodelf include them (it emits only terms that are zero in all rows)
        all_subsets = {
            subset
            for result in shapiq_results
            for subset in result.interactions
            if lowest <= len(subset) <= self._max_order
        }

        woodelf_format = {subset: np.zeros(len(shapiq_results)) for subset in all_subsets}

        # a subset a row does not report stays 0.0
        for row, result in enumerate(shapiq_results):
            for subset, value in result.interactions.items():
                if subset in woodelf_format:
                    woodelf_format[subset][row] = value

        return woodelf_format

    def _woodelf_result_to_batch(
        self, woodelf_result: dict[tuple, np.ndarray], n_players: int, n_instances: int
    ) -> InteractionValuesBatch:
        """Wrap a Woodelf result in an :class:`~shapiq.interaction_values.InteractionValuesBatch`.

        Args:
            woodelf_result: Woodelf-format values
                ``{interaction_tuple: ndarray of shape (n_instances,)}``.
            n_players: The number of features of the explained instances.
            n_instances: The number of explained instances.

        Returns:
            The batch carrying this explainer's index, orders, and baseline value.
        """
        return InteractionValuesBatch(
            woodelf_result,
            n_instances=n_instances,
            n_players=n_players,
            index=self._index,
            max_order=self._max_order,
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
            # add the baseline value to the empty prediction
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
                return self._woodelf_result_to_batch(
                    woodelf_explanation, n_players=int(x.shape[0]), n_instances=1
                )[0]

        if not self._explainers_initialized:
            self._init_explainers()
        if self.mode == "pathdependent":
            # dispatch on whichever per-tree list _init_explainers chose to populate.
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
    ) -> InteractionValuesBatch:
        """Explain multiple instances at once, using Woodelf on larger inputs.

        The whole batch is computed in a single vectorized Woodelf run when the input crosses
        the cut-offs (see :meth:`_should_use_woodelf`), and by the per-instance shapiq
        computation otherwise. Either way the result is an
        :class:`~shapiq.interaction_values.InteractionValuesBatch`: a sequence of one
        :class:`~shapiq.interaction_values.InteractionValues` per instance (materialized lazily
        on access), whose ``values`` attribute exposes the memory-efficient vectorized format
        ``{interaction_tuple: ndarray of shape (n_instances,)}`` directly.

        Args:
            X: A 2-dimensional matrix of inputs to be explained with shape
                ``(n_instances, n_features)``.
            n_jobs: Number of jobs for the shapiq fallback's ``joblib.Parallel``. Defaults to
                ``None`` (no parallelization).
            random_state: The random state to re-initialize the shapiq fallback with. Defaults to
                ``None``.
            verbose: Whether to print a progress bar in the shapiq fallback. Defaults to
                ``False``.
            **kwargs: Additional keyword-only arguments passed to the shapiq fallback.

        Returns:
            The interaction values of all instances in ``X`` as a batch.
        """
        n_players = int(X.shape[1])
        if self._should_use_woodelf(len(X)):
            woodelf_result = self._run_woodelf(X)
            if woodelf_result is not None:
                return self._woodelf_result_to_batch(
                    woodelf_result, n_players=n_players, n_instances=len(X)
                )

        shapiq_results = super().explain_X(
            X, n_jobs=n_jobs, random_state=random_state, verbose=verbose, **kwargs
        )
        # transposed into the vectorized layout so the batch is the same on both routes
        return self._woodelf_result_to_batch(
            self._cast_shapiq_results_to_woodelf_format(list(shapiq_results)),
            n_players=n_players,
            n_instances=len(X),
        )
