"""The GraphExplainer class for the shapiq package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import joblib
import numpy as np
from torch_geometric.data import Data
from tqdm.auto import tqdm

from shapiq.explainer.base import Explainer
from shapiq.graph.l_shapley import LShapley
from shapiq.interaction_values import InteractionValues
from shapiq_games.benchmark.graphshapiq_xai.base import GraphGame

from .graphshapiq import GraphSHAPIQ

if TYPE_CHECKING:
    from torch import nn

    from shapiq.explainer.custom_types import ExplainerIndices

SPARSIFY_THRESHOLD = 1e-8


class GraphExplainer(Explainer):
    """The GraphExplainer class for graph-based models.

    The explainer for graph-based models using the
    :class:`~shapiq.graph.graphshapiq.GraphSHAPIQ` algorithm. For details, refer to
    `Muschalik et al. (2025)` [Mus25]_.

    GraphSHAP-IQ is an algorithm for computing Shapley Interaction values for graph-based models.
    It is based on the GraphSHAPIQ algorithm by `Muschalik et al. (2025)` [Mus25]_, which efficiently
    calculates any order Shapley Interactions for GNN-based predictions, by utilizing the unique
    characteristics of GNNs, where only nodes in node i's l-hop neighborhood are considered for coalitions,
    thus drastically reducing the total amount of coalitions to evaluate.

    The GraphExplainer can be used with a variety of graph-based models, including
    GCNs, GATs and GINS.

    References:
        .. [Mus25] Maximilian Muschalik and Fabian Fumagalli and Paolo Frazzetto and Janine Strotherm and Luca Hermes and Alessandro Sperduti and Eyke Hüllermeier and Barbara Hammer. Exact Computation of Any-Order Shapley Interactions for Graph Neural Networks. https://arxiv.org/abs/2501.16944
    """

    def __init__(
        self,
        model: nn.Module,
        l_shapley_max_budget: int = 20000,
        index: ExplainerIndices = "k-SII",
        baseline_strategy: str = "max",
        max_order: int = 2,
        class_index: int | None = None,
        *,
        normalize: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the GraphExplainer.

        Args:
            model: A Graph-based model to explain.
            l_shapley_max_budget: Maximum budget for LShapley approximation.
            index: The type of Shapley interaction index to use. Defaults to "k-SII",
                which computes the k-Shapley Interaction Index. If max_order is set
                to 1, this corresponds to the Shapley value (index="SV"). Options are:
                "SV": Shapley value
                "k-SII": k-Shapley Interaction Index
                "FSII": Faithful Shapley Interaction Index
                "FBII": Faithful Banzhaf Interaction Index (becomes BV for order 1)
                "STII": Shapley Taylor Interaction Index
                "SII": Shapley Interaction Index
            class_index: The class index of the model to explain. Defaults to None,
                which will set the class index to 1 per default for classification
                models and is ignored for regression models. Note, it is important
                to specify the class index for your classification model.
            max_order: The maximum interaction order to be computed.
                Defaults to 2. Set to 1 for no interactions (single feature attribution)
            normalize: Whether to normalize graphshapiq algorithm.
            baseline_strategy: The node masking strategy.
            **kwargs: Additional keyword arguments are ignored.
        """
        super().__init__(model, class_index=class_index, index=index, max_order=max_order)  # type: ignore[arg-type]
        self._model: nn.Module = cast("nn.Module", self._model)
        self._class_index = class_index
        self._baseline_strategy = baseline_strategy
        self._normalize = normalize
        self._l_shapley_max_budget: int = l_shapley_max_budget

    @override
    def explain_X(
        self,
        X: np.ndarray | list[Data],
        *,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[InteractionValues]:
        """Explain multiple graph predictions at once.

        Args:
            X: A list of :class:`~torch_geometric.data.Data` objects to be explained.
                Passing a NumPy array (as accepted by the base class) is not supported
                and will raise a ``TypeError``.
            n_jobs: Number of jobs for ``joblib.Parallel``. Defaults to ``None`` (no
                parallelization). Set to ``-1`` to use all available cores.
            random_state: Unused; kept for API compatibility with the base class.
            verbose: Whether to show a progress bar. Defaults to ``False``.
            **kwargs: Additional keyword-only arguments forwarded to ``explain_function``.

        Returns:
            A list of interaction values, one per graph in ``X``.

        Raises:
            TypeError: If ``X`` is a NumPy array instead of a list of ``Data`` objects.
        """
        if isinstance(X, np.ndarray):
            msg = (
                "GraphExplainer.explain_X expects a list of torch_geometric.data.Data objects, "
                "not a NumPy array."
            )
            raise TypeError(msg)

        if n_jobs:
            parallel = joblib.Parallel(n_jobs=n_jobs)
            return cast(
                "list[InteractionValues]",
                parallel(joblib.delayed(self.explain)(x_graph, **kwargs) for x_graph in X),
            )
        pbar = tqdm(total=len(X), desc="Explaining") if verbose else None
        ivs: list[InteractionValues] = []
        for x_graph in X:
            ivs.append(self.explain(x_graph, **kwargs))
            if pbar is not None:
                pbar.update(1)
        return ivs

    @override
    def explain_function(
        self,
        x: Data,
        *,
        l_shapley: bool = False,
        max_interaction_size: int | None = None,
        **kwargs: Any,
    ) -> InteractionValues:
        """Computes the Shapley Interaction values for a single instance.

        Args:
            x: The input graph to explain.
            l_shapley: If ``True``, run the L-Shapley approximation; if ``False`` (default),
                run the exact GraphSHAP-IQ computation.
            max_interaction_size: Maximum k-hop neighbourhood size for the L-Shapley
                approximation.  When ``None`` the full neighbourhood size reported by
                :class:`~shapiq.graph.graphshapiq.GraphSHAPIQ` is used.  Ignored when
                *l_shapley* is ``False``.
            **kwargs: Allows for passing and overriding ``index`` argument.

        Returns:
            The interaction values for the instance.
        """
        index = kwargs.get("index", self._index)

        game = GraphGame(
            model=cast("nn.Module", self._model),
            x_graph=x,
            baseline_strategy=self._baseline_strategy,
            normalize=self._normalize,
            class_index=self._class_index,
            verbose=True,
        )
        explainer = GraphSHAPIQ(game=game)
        self._check_total_budget(explainer.total_budget)

        if l_shapley:
            effective_size = (
                max_interaction_size
                if max_interaction_size is not None
                else explainer.max_size_neighbors
            )
            return self._run_l_shapley_approximation(game, explainer, effective_size, index)
        return self._run_graph_shapiq_approximation(game, explainer, index)

    @override
    def explain(self, x: np.ndarray | Data | None = None, **kwargs: Any) -> InteractionValues:
        """Explain a single graph prediction.

        Args:
            x: The input graph to explain as a
                :class:`~torch_geometric.data.Data` object.
            **kwargs: Additional keyword-only arguments forwarded to ``explain_function``.

        Returns:
            The interaction values for the graph.
        """
        if not isinstance(x, Data):
            msg = f"GraphExplainer requires a torch_geometric.data.Data object, got {type(x).__name__!r}."
            raise TypeError(msg)
        return self.explain_function(x=x, **kwargs)

    def _run_graph_shapiq_approximation(
        self,
        game: GraphGame,
        explainer: GraphSHAPIQ,
        index: str = "k-SII",
    ) -> InteractionValues:
        """Approximate Shapley Interactions using grapshapiq."""
        moebius, _ = explainer.explain(
            max_interaction_size=explainer.max_size_neighbors,
            order=game.n_players,
            efficiency_routine=True,
            index=index,
        )

        if not isinstance(moebius, InteractionValues):
            err_msg = f"Expected InteractionValues, got {type(moebius)}"
            raise TypeError(err_msg)
        moebius.estimation_budget = explainer.last_n_model_calls
        moebius.estimated = False
        moebius.sparsify(threshold=SPARSIFY_THRESHOLD)

        return moebius

    def _run_l_shapley_approximation(
        self,
        game: GraphGame,
        explainer: GraphSHAPIQ,
        max_interaction_size: int,
        index: str,
    ) -> InteractionValues:
        """Run the L-Shapley approximation.

        Args:
            game: The constructed graph game for this instance.
            explainer: The GraphSHAPIQ explainer initialised from the game.
            max_interaction_size: Maximum k-hop neighbourhood size to consider.  This value
                comes from the caller and is passed directly into
                :meth:`~shapiq.graph.l_shapley.LShapley.explain` so that
                ``LShapley.max_size_neighbors`` (which is ``0`` before ``explain()`` runs)
                is never used as the input.
            index: The type of the interaction values.

        Returns:
            The approximated Shapley values as an
            :class:`~shapiq.interaction_values.InteractionValues` object.
        """
        l_shapley_explainer = LShapley(game, max_budget=explainer.total_budget)

        shapley_values, _ = l_shapley_explainer.explain(
            max_interaction_size=max_interaction_size, break_on_exceeding_budget=False, index=index
        )

        shapley_values.estimation_budget = l_shapley_explainer.last_n_model_calls
        shapley_values.estimated = True
        shapley_values.sparsify(threshold=SPARSIFY_THRESHOLD)

        return shapley_values

    def _check_total_budget(self, total_budget: int) -> None:
        """Check if total_budget is within the max budget."""
        if total_budget > self._l_shapley_max_budget:
            msg = (
                f"Total budget of {total_budget} exceeds the limit of {self._l_shapley_max_budget}."
            )
            raise RuntimeError(msg)
