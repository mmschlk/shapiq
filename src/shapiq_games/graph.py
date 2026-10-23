"""The game of a graph neural network's output for one graph with nodes removed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq.graph import (
    GraphBaselineImputer,
    GraphModel,
    convert_graph,
    get_graph_predict_function,
)
from shapiq.graph.conversion import resolve_n_hops
from shapiq_games._base import as_bool_coalitions

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from shapiq.typing import CoalitionMatrix, GameValues

__all__ = ["GraphGame"]


class GraphGame(Game):
    """The output of a graph neural network for one graph, given a coalition of its nodes.

    The players are the nodes of the graph. The :class:`~shapiq.graph.GraphBaselineImputer`
    replaces the node features of absent nodes by a baseline while keeping the edges
    :cite:p:`Muschalik.2025`: the mean of the background node features ``data``, or zeros. The
    value of a coalition is the graph-level output of the model for the imputed graph: the
    explained class's logit for a model with several outputs, else its single output.

    The game is local on the graph if the model consists of ``n_hops`` message passing layers
    followed by a linear global pooling (e.g. a sum or mean over the nodes) and a linear readout,
    and :class:`~shapiq.graph.GraphSHAPIQ` computes its values exactly. A softmax or a nonlinear
    readout after the pooling makes the game non-local, so explain logits. A ``torch`` model is
    evaluated on padded batches of ``batch_size`` imputed graphs (see
    :func:`~shapiq.graph.get_graph_predict_function`), so the value of a coalition does not depend
    on which coalitions are evaluated with it.

    Attributes:
        model: The graph neural network.
        graph: The explained graph as a :class:`~shapiq.graph.GraphModel`.
        class_index: The explained output for models with several outputs, ``None`` for models
            with one output.
        imputer: The imputer turning the model into a game.
        empty_value: The output with all nodes absent (before centering).
        original_model_output: The output for the graph (before centering).

    Examples:
        >>> import torch
        >>> from torch_geometric.data import Data
        >>> from torch_geometric.nn import GCNConv, global_mean_pool
        >>> class GCN(torch.nn.Module):
        ...     num_layers = 1
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv, self.readout = GCNConv(3, 8), torch.nn.Linear(8, 2)
        ...     def forward(self, x, edge_index, batch):
        ...         return self.readout(global_mean_pool(self.conv(x, edge_index).relu(), batch))
        >>> graph = Data(x=torch.rand(4, 3), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        >>> game = GraphGame(GCN(), graph)  # explains the logit of class 1
        >>> game.n_players, game.graph.n_hops
        (4, 1)
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        graph: Data | GraphModel,
        *,
        data: np.ndarray | None = None,
        class_index: int | None = None,
        n_hops: int | None = None,
        batch_size: int = 128,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the game.

        Args:
            model: A graph neural network with a graph-level output, called as
                ``model(x=x, edge_index=edge_index, batch=batch)``, or a function of node features
                of shape ``(n_graphs, n_nodes, n_features)`` for a graph given as
                :class:`~shapiq.graph.GraphModel`.
            graph: The explained graph with node features, a ``torch_geometric.data.Data`` graph
                or a :class:`~shapiq.graph.GraphModel`.
            data: The background node features of shape ``(n_samples, n_features)``, whose mean is
                the baseline of absent nodes, or the baseline node features of shape
                ``(n_features,)``. Defaults to ``None``, which means zeros.
            class_index: The explained output of a model with several outputs. Defaults to
                ``None``, which means output ``1`` for models with several outputs (the convention
                of the shapiq explainers) and the output of models with one output.
            n_hops: The number of message passing layers of the model. Defaults to ``None``,
                which takes ``model.num_layers`` (or ``n_hops`` of a
                :class:`~shapiq.graph.GraphModel`).
            batch_size: The number of graphs per forward pass of a ``torch`` model. Defaults to
                ``128``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        if not isinstance(graph, GraphModel):
            n_hops = resolve_n_hops(model, n_hops)
        self.model = model
        self.graph = convert_graph(graph, n_hops=n_hops)
        predict = get_graph_predict_function(
            model, self.graph, class_index=class_index, batch_size=batch_size
        )
        self.class_index = predict.class_index
        background = np.zeros(self.graph.n_features) if data is None else np.asarray(data)
        self.imputer = GraphBaselineImputer(predict, background, x=self.graph, normalize=False)
        self.empty_value = self.imputer.empty_prediction
        n_nodes = self.imputer.n_players
        self.original_model_output = float(
            self.value_function(np.ones((1, n_nodes), dtype=bool))[0]
        )
        super().__init__(
            n_nodes,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the model's output for the graph with the absent nodes imputed."""
        return self.imputer.value_function(as_bool_coalitions(coalitions))
