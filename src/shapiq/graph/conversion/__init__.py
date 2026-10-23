"""Conversion of the graphs and models of graph libraries for GraphSHAP-IQ.

:func:`convert_graph` turns a graph into a :class:`~shapiq.graph.GraphModel`, and
:func:`get_graph_predict_function` turns a model into a function of the node features of masked
copies of that graph, which the :class:`~shapiq.graph.GraphBaselineImputer` calls. Supported are
``torch_geometric`` graphs and ``torch`` models called as ``model(x=x, edge_index=edge_index,
batch=batch)``, and :class:`~shapiq.graph.GraphModel` graphs with any function of node features of
shape ``(n_graphs, n_nodes, n_features)``.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.graph.base import GraphModel
from shapiq.utils.modules import safe_isinstance

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "GraphPredictFunction",
    "convert_graph",
    "get_graph_predict_function",
    "require_torch_geometric",
    "resolve_n_hops",
]


def require_torch_geometric() -> None:
    """Raise an ``ImportError`` naming the ``graph`` extra if torch or torch_geometric is missing.

    Raises:
        ImportError: If ``torch`` or ``torch_geometric`` cannot be imported.
    """
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as error:
        msg = (
            "Graphs of torch_geometric require torch and torch_geometric. Install them with: "
            "pip install 'shapiq[graph]'"
        )
        raise ImportError(msg) from error


def resolve_n_hops(model: object, n_hops: int | None) -> int:
    """Return ``n_hops``, or the ``num_layers`` of the model (set by ``torch_geometric.nn``).

    Raises:
        ValueError: If ``n_hops`` is not given and the model has no integer ``num_layers``.
    """
    if n_hops is None:
        n_hops = getattr(model, "num_layers", None)
    if isinstance(n_hops, bool) or not isinstance(n_hops, int | np.integer) or n_hops < 0:
        msg = (
            "Pass n_hops, the number of message passing layers of the model: it has no "
            f"non-negative integer num_layers attribute, got n_hops={n_hops!r}."
        )
        raise ValueError(msg)
    return int(n_hops)


def convert_graph(graph: Any, *, n_hops: int | None = None) -> GraphModel:  # noqa: ANN401
    """Convert a graph into a :class:`~shapiq.graph.GraphModel`.

    Args:
        graph: A :class:`~shapiq.graph.GraphModel` or a ``torch_geometric.data.Data`` graph with
            node features ``x`` and edges ``edge_index``.
        n_hops: The radius of the receptive fields. Required for a ``torch_geometric`` graph;
            replaces the one of a :class:`~shapiq.graph.GraphModel` if given.

    Returns:
        The graph.

    Raises:
        TypeError: If the graph is of an unsupported type.
        ValueError: If ``n_hops`` is missing or the graph has no node features or edges.
    """
    if isinstance(graph, GraphModel):
        if n_hops is None or n_hops == graph.n_hops:
            return graph
        return dataclasses.replace(graph, n_hops=n_hops)
    if safe_isinstance(graph, "torch_geometric.data.Data"):
        if n_hops is None:
            msg = "Pass n_hops, the number of message passing layers, for a torch_geometric graph."
            raise ValueError(msg)
        if graph.x is None or graph.edge_index is None:
            msg = "The graph needs node features (graph.x) and edges (graph.edge_index)."
            raise ValueError(msg)
        return GraphModel(
            edge_index=graph.edge_index.detach().cpu().numpy(),
            n_hops=n_hops,
            x=graph.x.detach().cpu().numpy(),
            n_nodes=int(graph.num_nodes),
        )
    msg = f"Graphs of type {type(graph).__name__} are not supported."
    raise TypeError(msg)


class GraphPredictFunction(ABC):
    """A model as a function of the node features of masked copies of one graph.

    Calling it on node features of shape ``(n_graphs, n_nodes, n_features)`` returns the explained
    output for every copy: the output ``class_index`` of a model with several outputs, else its
    only output.

    Attributes:
        graph: The graph whose edges every copy keeps.
        class_index: The explained output for models with several outputs, ``None`` for models
            with one output.
    """

    def __init__(self, graph: GraphModel, class_index: int | None) -> None:
        """Bind the function to the graph and resolve the explained output on its node features.

        Raises:
            ValueError: If the graph has no node features or the class index is out of range.
        """
        if graph.x is None:
            msg = "The graph needs node features (x) to call the model on."
            raise ValueError(msg)
        self.graph = graph
        n_outputs = self.outputs(graph.x[None]).shape[1]
        self.class_index = _resolve_class_index(class_index, n_outputs)

    @abstractmethod
    def outputs(self, node_features: np.ndarray) -> np.ndarray:
        """Return all outputs of shape ``(n_graphs, n_outputs)`` for the node features."""

    def __call__(self, node_features: np.ndarray) -> np.ndarray:
        """Return the explained output for the node features of every copy of the graph."""
        return self.outputs(node_features)[:, 0 if self.class_index is None else self.class_index]


class _CallablePredictFunction(GraphPredictFunction):
    """A function of node features of shape ``(n_graphs, n_nodes, n_features)``."""

    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        graph: GraphModel,
        class_index: int | None,
    ) -> None:
        self.function = function
        super().__init__(graph, class_index)

    def outputs(self, node_features: np.ndarray) -> np.ndarray:
        outputs = np.asarray(self.function(node_features), dtype=float)
        return outputs.reshape(len(node_features), -1)


def get_graph_predict_function(
    model: Any,  # noqa: ANN401
    graph: GraphModel,
    *,
    class_index: int | None = None,
    batch_size: int = 128,
) -> GraphPredictFunction:
    """Return the model as a function of the node features of masked copies of the graph.

    Args:
        model: A ``torch`` model called as ``model(x=x, edge_index=edge_index, batch=batch)`` that
            returns one output per graph of the batch (e.g. a graph neural network of
            ``torch_geometric`` with a global pooling), or a function of node features of shape
            ``(n_graphs, n_nodes, n_features)``.
        graph: The graph.
        class_index: The explained output of a model with several outputs. Defaults to ``None``,
            which means output ``1`` for models with several outputs (the convention of the shapiq
            explainers) and the output of models with one output.
        batch_size: The number of graphs per forward pass of a ``torch`` model. Defaults to
            ``128``.

    Returns:
        The function.

    Raises:
        TypeError: If the model is neither a ``torch`` model nor callable.
    """
    if safe_isinstance(model, "torch.nn.Module"):
        require_torch_geometric()
        from .torch_geometric import TorchGeometricPredictFunction

        return TorchGeometricPredictFunction(
            model, graph, class_index=class_index, batch_size=batch_size
        )
    if callable(model):
        return _CallablePredictFunction(model, graph, class_index)
    msg = f"Models of type {type(model).__name__} are not supported."
    raise TypeError(msg)


def _resolve_class_index(class_index: int | None, n_outputs: int) -> int | None:
    """Return the explained output: ``None`` for one output, class ``1`` by default."""
    if n_outputs == 1:
        if class_index not in (None, 0):
            msg = f"The model has one output, got class_index={class_index}."
            raise ValueError(msg)
        return None
    if class_index is None:
        return 1
    if not 0 <= class_index < n_outputs:
        msg = f"class_index must be in 0, ..., {n_outputs - 1}, got {class_index}."
        raise ValueError(msg)
    return class_index
