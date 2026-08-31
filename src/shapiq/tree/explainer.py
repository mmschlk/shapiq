"""Implementation of the TreeExplainer class.

The :class:`~shapiq.tree.explainer.TreeExplainer` computes path-dependent explanations with
the :class:`~shapiq.tree.quadrature.QuadratureTreeSHAP` algorithm and interventional
explanations with :class:`~shapiq.tree.interventional.computer.InterventionalTreeSHAPIQ`,
routing large interventional inputs through the optional Woodelf fast path.
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from shapiq.explainer.base import Explainer
from shapiq.interaction_values import InteractionValues, InteractionValuesBatch
from shapiq.tree.interventional.computer import InterventionalTreeSHAPIQ
from shapiq.utils.modules import safe_isinstance

from .base import TreeModel
from .quadrature import QuadratureTreeSHAP, QuadratureTreeSHAPIndices
from .validation import validate_tree_model

if TYPE_CHECKING:
    from shapiq.explainer.custom_types import ExplainerIndices
    from shapiq.typing import Model

TREE_MODES = Literal["pathdependent", "interventional"]
TREE_BACKENDS = Literal["auto", "woodelf", "shapiq"]
TreeExplainerIndices = Literal["SV", "SII", "k-SII", "BV", "BII", "STII", "FSII", "FBII", "Moebius"]

_WOODELF_INSTALL_HINT = "Install it with: pip install shapiq[tree]"
_WOODELF_REQUIRED = f"requires the optional 'woodelf-explainer' package. {_WOODELF_INSTALL_HINT}"

_WOODELF_INTERVENTIONAL_CUTOFF = 100_000
"""Interventional inputs with ``n_explained * n_reference`` at or above this route to Woodelf.

See :meth:`TreeExplainer._should_use_woodelf` for the measurements behind the value.
"""


class WoodelfNotAvailableWarning(UserWarning):
    """The explanation would be computed faster with the optional woodelf dependency.

    Emitted when an input crosses :class:`TreeExplainer`'s interventional Woodelf cut-off but
    the optional ``woodelf-explainer`` package is not installed, so the (slower) shapiq
    implementation computes the explanation instead. Filter it with
    ``warnings.filterwarnings("ignore", category=WoodelfNotAvailableWarning)`` or silence it
    for good by installing ``shapiq[tree]``.
    """


class TreeExplainer(Explainer):
    """The TreeExplainer class for tree-based models.

    The ``TreeExplainer`` is the model-specific explainer for tree-based models, capable of computing attributions and interactions
    for both path-dependent and interventional modes (details below).
    It supports various interaction indices and can leverage the optional Woodelf package for efficient computation on large datasets.
    We support the following model types: ``scikit-learn`` decision trees, random forests, and gradient-boosted ensembles, as well as ``XGBoost``, ``LightGBM``, and ``CatBoost`` models, for both regression and classification tasks.
    The model may be fitted both on categorical and numerical features (except for CatBoost, which requires categorical features to be encoded as integers).
    Attributions and interactions are returned as :class:`~shapiq.interaction_values.InteractionValues` objects. That object is
    indexed by feature tuples — ``values[(1,)]`` is the attribution of feature 1,
    ``values[(1, 2)]`` the interaction of features 1 and 2 — carries the expected model
    output as ``baseline_value``, and offers plotting shortcuts such as
    :meth:`~shapiq.interaction_values.InteractionValues.plot_force` and
    :meth:`~shapiq.interaction_values.InteractionValues.plot_network`.

    :meth:`explain_X` explains a whole data matrix at once, where we return a :class:`~shapiq.interaction_values.InteractionValuesBatch` object, for which ``values[(1,)]`` is a 1D array of the attributions of feature 1 for all explained instances, and ``values[(1, 2)]`` is a 1D array of the interactions of features 1 and 2 for all explained instances.

    Two established tree-explanation modes :cite:t:`Lundberg.2020` differ in what "feature
    :math:`i` is absent" means:

    - In ``"pathdependent"`` mode (the default) an absent feature follows *both* children of
      its split nodes, weighted by the share of training samples that went each way — the
      background distribution is the one already stored in the tree, so no data is needed.
      The computation uses Quadrature-TreeSHAP :cite:t:`Wettenstein.2026a` (whose first-order
      case was independently derived in TreeGrad-Ranker :cite:t:`Li.2026`), implemented in
      :class:`~shapiq.tree.quadrature.QuadratureTreeSHAP`: values and interactions are
      Gauss-Legendre integrals of weighted Banzhaf interaction polynomials, numerically exact
      in float64 at any tree depth. The algorithm descends from Linear TreeSHAP
      :cite:t:`Yu.2022` and computes any-order Shapley interactions as introduced for trees
      by TreeSHAP-IQ :cite:t:`Muschalik.2024a`; Banzhaf indices fall out of the same
      polynomials by evaluation at participation probability 1/2, and the Moebius
      coefficients by evaluation at 0. Supported indices:
      ``"SV"``, ``"SII"``, ``"k-SII"``, ``"BV"``, ``"BII"``, and ``"Moebius"``.

    - In ``"interventional"`` mode an absent feature takes the values it has in a
      ``reference_dataset`` (background SHAP), computed by
      :class:`~shapiq.tree.interventional.computer.InterventionalTreeSHAPIQ`, which
      additionally supports the ``"STII"``, ``"FSII"``, and ``"FBII"`` indices. Large
      interventional inputs are routed to the vectorized Woodelf and WOODELF-HD algorithms
      :cite:t:`Nadel.2026` :cite:t:`Wettenstein.2026b` when the optional
      ``woodelf-explainer`` package is installed (``pip install shapiq[tree]``); the
      ``backend`` parameter overrides this routing.

    The two computation classes (:class:`~shapiq.tree.quadrature.QuadratureTreeSHAP` and
    :class:`~shapiq.tree.interventional.computer.InterventionalTreeSHAPIQ`) live in
    :mod:`shapiq.tree` and can be used directly, as can the standalone reference algorithms
    :class:`~shapiq.tree.treeshapiq.TreeSHAPIQ` and
    :class:`~shapiq.tree.linear.computer.LinearTreeSHAP`.

    Examples:
        Shapley values for a random forest — the attributions sum to the prediction
        (efficiency):

        >>> import numpy as np
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from shapiq import TreeExplainer
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(500, 5))
        >>> y = X[:, 0] + 2 * X[:, 1] * X[:, 2]
        >>> model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
        >>> explainer = TreeExplainer(model, index="SV", max_order=1)
        >>> shapley_values = explainer.explain(X[0])
        >>> shapley_values[(1,)]  # contribution of feature 1 to this prediction
        -0.0993...
        >>> float(shapley_values.values.sum())  # equals model.predict(X[:1])[0]
        -0.2279...

        Pairwise Shapley interactions (``k-SII`` of order 2) separate the learned
        ``X1 * X2`` synergy from the additive effects:

        >>> explainer = TreeExplainer(model, index="k-SII", max_order=2)
        >>> interactions = explainer.explain(X[0])
        >>> abs(interactions[(1, 2)]) > 10 * abs(interactions[(0, 1)])
        True

        Interventional Shapley values against a background dataset:

        >>> explainer = TreeExplainer(
        ...     model,
        ...     mode="interventional",
        ...     reference_dataset=X[:100],
        ... )
        >>> shapley_values = explainer.explain(X[0])

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
                In ``"pathdependent"`` mode, the explainer computes path-dependent interaction
                values with the Quadrature-TreeSHAP algorithm; in ``"interventional"`` mode, it
                computes interventional interaction values against the ``reference_dataset``.
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

            index: The type of interaction to be computed. In ``"pathdependent"`` mode, the
                indices ``["SV", "SII", "k-SII", "BV", "BII", "Moebius"]`` are supported. In
                ``"interventional"`` mode, further indices such as ``"STII"``, ``"FSII"``, or
                ``"FBII"`` can be computed. Defaults to ``"SV"``.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models.

            reference_dataset: A dataset to be used for reference in the explanation. Required
                when ``mode="interventional"``. Defaults to ``None``.

            backend: Which implementation computes the explanations. With ``"auto"`` the
                explainer computes path-dependent explanations with shapiq's
                :class:`~shapiq.tree.quadrature.QuadratureTreeSHAP` and routes larger
                interventional inputs to Woodelf (falling back to shapiq with a
                :class:`WoodelfNotAvailableWarning` if the optional woodelf dependency is
                missing). ``"woodelf"`` forces Woodelf and raises if it is not installed or
                cannot handle the configuration; ``"shapiq"`` forces the shapiq
                implementation. Defaults to ``"auto"``.

            **kwargs: Additional keyword arguments are ignored.

        """
        # "Moebius" is only computed by TreeExplainer, so it is not part of the shared
        # ``ExplainerIndices`` the base class (and the approximators) are typed with.
        super().__init__(model, index=cast("ExplainerIndices", index), max_order=max_order)

        if min_order < 0 or min_order > self._max_order:
            msg = (
                f"min_order={min_order} must satisfy 0 <= min_order <= max_order "
                f"(max_order={self._max_order})."
            )
            raise ValueError(msg)

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
        self._mode: TREE_MODES = mode
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
        if mode == "pathdependent" and self.index not in (
            "SV",
            "SII",
            "k-SII",
            "BV",
            "BII",
            "Moebius",
        ):
            msg = (
                f"index='{self.index}' is not supported in 'pathdependent' mode; use "
                "mode='interventional' with a reference_dataset."
            )
            raise ValueError(msg)
        if mode == "interventional" and self.index == "Moebius":
            # the interventional kernel has no Moebius leaf weight yet; only the
            # path-dependent quadrature kernel computes this index.
            msg = (
                "index='Moebius' is not supported in 'interventional' mode; use "
                "mode='pathdependent'."
            )
            raise ValueError(msg)

        self._pathdependent_explainer: QuadratureTreeSHAP | None = None
        self._interventional_explainer: InterventionalTreeSHAPIQ | None = None
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
                self._baseline_value = InterventionalTreeSHAPIQ.compute_empty_prediction(
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
        if self.mode == "pathdependent":
            self._pathdependent_explainer = QuadratureTreeSHAP(
                model=self._trees,
                max_order=self._max_order,
                index=cast("QuadratureTreeSHAPIndices", self.index),
            )
        elif self.mode == "interventional":
            if self._reference_dataset is None:
                msg = (
                    "InterventionalTreeSHAPIQ requires a reference_dataset; pass one to "
                    "TreeExplainer(..., mode='interventional', reference_dataset=...)."
                )
                raise ValueError(msg)
            self._interventional_explainer = InterventionalTreeSHAPIQ(
                model=self._trees,
                data=self._reference_dataset,
                class_index=self._class_label,
                max_order=self._max_order,
                index=self.index,
            )
        # only mark initialized once the explainer was fully built: a constructor that raises
        # must re-raise on retry, not leave the explainer half-built
        self._explainers_initialized = True

    def _should_use_woodelf(self, number_of_explained_instances: int) -> bool:
        """Decide whether Woodelf or the shapiq implementation computes the explanation.

        Path-dependent explanations are computed by the shapiq quadrature kernel (unless
        ``backend="woodelf"`` forces Woodelf). In interventional mode the cut-off is
        ``n * m >= 100_000`` (:data:`_WOODELF_INTERVENTIONAL_CUTOFF`), where ``n`` is the
        number of explained instances and ``m`` is the size of the reference dataset.

        The value was re-measured after the improvements to
        :class:`~shapiq.tree.interventional.computer.InterventionalTreeSHAPIQ` in
        `PR #590 <https://github.com/mmschlk/shapiq/pull/590>`_ sped it up by ~2
        orders of magnitude (the previous cut-off of ``100`` predates that). End-to-end
        timings (explainer construction + ``explain_X``) of both backends over sklearn
        random forests and XGBoost ensembles (50-300 trees, depth 3-8, 10-50 features,
        400-25k leaves) for ``"SV"`` and order-2 ``"k-SII"`` put the break-even between
        ``n * m ~= 1e4`` (small ensembles, where Woodelf's fixed per-tree preprocessing
        cost is also tiny) and ``n * m ~= 1e6`` (large forests). The penalty is
        asymmetric: below the break-even, Woodelf's fixed preprocessing cost makes it up
        to hundreds of times slower, while above it shapiq's per-instance loop trails by
        a low single-digit factor near the boundary; ``1e5`` balances the worst-case
        absolute losses across the measured model shapes and ``n``/``m`` splits.

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
        if self.mode != "interventional":
            return False

        if self._woodelf_unsupported_reason() is not None:
            return False

        if (
            self._reference_dataset is not None
            and len(self._reference_dataset) * number_of_explained_instances
            >= _WOODELF_INTERVENTIONAL_CUTOFF
        ):
            return self._woodelf_available()
        return False

    def _woodelf_unsupported_reason(self) -> str | None:
        """The reason why Woodelf cannot serve this explainer's configuration, if any.

        Returns:
            A human-readable reason, or ``None`` when Woodelf supports the configuration.
        """
        if self.index not in (
            "SV",
            "BV",
            "SII",
            "k-SII",
            "BII",
            "STII",
            "FSII",
            "FBII",
            "Moebius",
        ):
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

    def _run_woodelf(self, X: np.ndarray) -> dict[tuple, np.ndarray]:
        """Compute Shapley or Banzhaf (interaction) values with Woodelf's ``hybrid_woodelf``.

        Args:
            X: The instances to explain as a 2-dimensional array of shape
                ``(n_instances, n_features)``.

        Returns:
            The values in the Woodelf format ``{interaction_tuple: ndarray of shape
            (n_instances,)}``.

        Raises:
            ValueError: If the configuration is not supported by Woodelf (only reachable
                with ``backend="woodelf"``; auto routing checks the configuration first).
        """
        try:
            import pandas as pd
            from woodelf.core.cube_metric import (
                BanzhafValues,
                FaithfulBanzhafInteractionValues,
                FaithfulShapleyInteractionValues,
                GeneralBanzhafInteractionValues,
                GeneralShapleyInteractionValues,
                MobiusCoefficients,
                ShapleyTaylorInteractionValues,
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

        index_to_metric_class = {
            "SII": GeneralShapleyInteractionValues,
            "k-SII": GeneralShapleyInteractionValues,  # k-SII is a pure aggregation of SII, Woodelf computes the SII base values and
            # ``_aggregate_batched_sii_to_ksii`` makes them k-SII
            "BII": GeneralBanzhafInteractionValues,
            "STII": ShapleyTaylorInteractionValues,
            "FSII": FaithfulShapleyInteractionValues,
            "FBII": FaithfulBanzhafInteractionValues,
            "Moebius": MobiusCoefficients,
        }
        if self._index == "SV":
            metric = ShapleyValues()
        elif self._index == "BV":
            metric = BanzhafValues()
        elif self._index in index_to_metric_class:
            metric_class = index_to_metric_class[self._index]
            metric = metric_class(max(self._min_order, 1), self._max_order)
        else:  # pre-validated in __init__ / _woodelf_unsupported_reason; defensive
            msg = f"index='{self._index}' is not supported by Woodelf."
            raise ValueError(msg)

        class_index = self._class_label if self._class_label is not None else 1
        loaded_model = load_decision_tree_ensemble_model(
            self.model, range(X.shape[1]), class_index=class_index
        )

        # woodelf cannot compute path-dependent SHAP of order >= 3 on trees deeper than 16
        # yet; only a forced backend='woodelf' routes path-dependent explanations here.
        if self.mode == "pathdependent" and self.max_order >= 3 and loaded_model.max_depth > 16:
            msg = (
                "backend='woodelf' cannot compute path-dependent interactions of "
                f"order >= 3 on trees deeper than 16 (tree depth: "
                f"{loaded_model.max_depth}); use backend='auto' or backend='shapiq'."
            )
            raise ValueError(msg)

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
        # the isinstance check drops the scalar ``()`` baseline entry
        return {
            subset: values
            for subset, values in aggregated.items()
            if isinstance(values, np.ndarray) and lowest <= len(subset) <= self._max_order
        }

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

    def _explain_function_pathdependent(
        self,
        x: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Compute the explanation via the path-dependent quadrature explainer.

        The :class:`~shapiq.tree.quadrature.QuadratureTreeSHAP` built in ``_init_explainers``
        validates the model and aggregates over ensembles itself; this method only enforces
        the requested ``min_order`` on the result.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            The interaction values for the instance.
        """
        if len(x.shape) != 1:
            msg = "explain expects a single instance, not a batch."
            raise TypeError(msg)
        if self._pathdependent_explainer is None:
            msg = "Path-dependent explainer is not initialized; mode must be 'pathdependent'."
            raise RuntimeError(msg)

        final_explanation = self._pathdependent_explainer.explain(x)

        if self._min_order == 0 and final_explanation.min_order == 1:
            final_explanation.min_order = 0
            final_explanation.interactions[()] = float(final_explanation.baseline_value)

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
        """Compute interaction values for ``x`` via the eagerly-built :class:`InterventionalTreeSHAPIQ`.

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
            return self._woodelf_result_to_batch(
                woodelf_explanation, n_players=int(x.shape[0]), n_instances=1
            )[0]

        if not self._explainers_initialized:
            self._init_explainers()
        if self.mode == "pathdependent":
            return self._explain_function_pathdependent(x, **kwargs)
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
        """Explain multiple instances at once, using Woodelf on larger interventional inputs.

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
            return self._woodelf_result_to_batch(
                woodelf_result, n_players=n_players, n_instances=len(X)
            )

        # initialize before joblib pickles this object, so workers don't re-construct
        # the explainer (or re-emit construction-time warnings) once per task
        if not self._explainers_initialized:
            self._init_explainers()
        shapiq_results = super().explain_X(
            X, n_jobs=n_jobs, random_state=random_state, verbose=verbose, **kwargs
        )
        return InteractionValuesBatch.from_interaction_values(shapiq_results)
