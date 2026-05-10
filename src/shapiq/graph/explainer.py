"""Docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from shapiq.explainer.base import Explainer
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
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the GraphExplainer.

        Args:
            model (nn.Module): The GNN model used for prediction.
            x_graph (Data): The sample to predict
            kwargs: ignored

        """
        super().__init__(model)
        self._gnns: list[GraphModel] = validate_graph_model(model)
        self._n_gnns = len(self._gnns)
        self.game = GraphGame(model, x_graph=x_graph)
        self.explainer = GraphSHAPIQ(game=self.game)

    @override
    def explain_function(
        self,
        x: np.ndarray | None,
        *,
        l_shapley: bool = False,
        **kwargs: Any,
    ) -> InteractionValues:
        """Run explain function."""
        if x is not None:
            msg = "GraphExplainer does not use x; graph data is set at init time."
            raise ValueError(msg)

        total_budget = self.explainer.total_budget
        if total_budget > MAX_BUDGET:
            msg = "Total budget higher than the limit."
            raise RuntimeError(msg)

        moebius, _ = self.explainer.explain(
            max_interaction_size=self.explainer.max_size_neighbors,
            order=self.game.n_players,
            efficiency_routine=True,
        )
        moebius.estimation_budget = self.explainer.last_n_model_calls
        moebius.estimated = False
        moebius.sparsify(threshold=1e-8)
        return moebius
