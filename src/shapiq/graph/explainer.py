"""Docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from shapiq.explainer.base import Explainer
from shapiq.graph.l_shapley import LShapley
from shapiq.graph.validation import validate_graph_model
from shapiq_games.benchmark.graphshapiq_xai.base import GraphGame

from .graphshapiq import GraphSHAPIQ

if TYPE_CHECKING:
    import numpy as np
    from torch import nn
    from torch_geometric.data import Data

    from shapiq.graph.base import GraphModel
    from shapiq.interaction_values import InteractionValues

MAX_BUDGET = 20_000
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
        x_graph: Data,
        l_shapley_max_budget: int = 20000,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the GraphExplainer.

        Args:
            model: A Graph-based model to explain.
            x_graph: The sample to explain.
            l_shapley_max_budget: Maximum budget for LShapley approximation.
            **kwargs: Additional keyword arguments are ignored.
        """
        super().__init__(model)
        self._gnns: list[GraphModel] = validate_graph_model(model)
        self._n_gnns = len(self._gnns)
        self.game = GraphGame(
            model=model,
            x_graph=x_graph,
            baseline_strategy="max",
            normalize=False,
            class_index=None,
            verbose=True,
        )
        self.l_shapley_max_budget = l_shapley_max_budget
        self.explainer = GraphSHAPIQ(game=self.game)
        self._check_total_budget()
        self.approx_explainer = None

    @override
    def explain_function(
        self,
        x: np.ndarray | None = None,
        *,
        l_shapley: bool = False,
        max_interaction_size: int | None = None,
        **kwargs: Any,
    ) -> InteractionValues:
        """Computes the Shapley Interaction values for a single instance.

        Args:
            x: The input graph.
            l_shapley: If ``True``, run the L-Shapley approximation; if ``False`` (default),
                run the exact GraphSHAP-IQ computation.
            max_interaction_size: Maximum k-hop neighbourhood size for the L-Shapley
                approximation.  When ``None`` the full neighbourhood size reported by
                :class:`~shapiq.graph.graphshapiq.GraphSHAPIQ` is used.  Ignored when
                *l_shapley* is ``False``.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            The interaction values for the instance.
        """
        if l_shapley:
            # Fall back to the GraphSHAPIQ neighbourhood size when the caller does not
            # supply an explicit limit.
            effective_size = (
                max_interaction_size
                if max_interaction_size is not None
                else self.explainer.max_size_neighbors
            )
            return self._run_l_shapley_approximation(effective_size)
        return self._run_graph_shapiq_approximation()

    def _run_graph_shapiq_approximation(self) -> InteractionValues:
        moebius, _ = self.explainer.explain(
            max_interaction_size=self.explainer.max_size_neighbors,
            order=self.game.n_players,
            efficiency_routine=True,
        )

        moebius.estimation_budget = self.explainer.last_n_model_calls
        moebius.estimated = False  # exact by definition
        moebius.sparsify(threshold=SPARSIFY_THRESHOLD)

        return moebius

    def _run_l_shapley_approximation(self, max_interaction_size: int) -> InteractionValues:
        """Run the L-Shapley approximation.

        Args:
            max_interaction_size: Maximum k-hop neighbourhood size to consider.  This value
                comes from the caller and is passed directly into
                :meth:`~shapiq.graph.l_shapley.LShapley.explain` so that
                ``LShapley.max_size_neighbors`` (which is ``0`` before ``explain()`` runs)
                is never used as the input.

        Returns:
            The approximated Shapley values as an :class:`~shapiq.interaction_values.InteractionValues`
            object.
        """
        l_shapley_explainer = LShapley(self.game, max_budget=self.explainer.total_budget)

        shapley_values, _ = l_shapley_explainer.explain(
            max_interaction_size=max_interaction_size,
            break_on_exceeding_budget=False,
        )

        shapley_values.estimation_budget = l_shapley_explainer.last_n_model_calls
        shapley_values.estimated = True  # always approximate for L-Shapley
        shapley_values.sparsify(threshold=SPARSIFY_THRESHOLD)

        return shapley_values

    def _check_total_budget(self) -> None:
        total_budget = self.explainer.total_budget
        if total_budget > self.l_shapley_max_budget:
            msg = f"Total budget of {total_budget} exceeds the limit of {MAX_BUDGET}."
            raise RuntimeError(msg)
