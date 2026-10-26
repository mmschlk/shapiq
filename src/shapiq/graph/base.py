"""The internal representation of a graph for GraphSHAP-IQ."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import sparse

__all__ = ["GraphModel"]


@dataclass(frozen=True, eq=False)
class GraphModel:
    """A graph as a graph neural network with ``n_hops`` message passing layers sees it.

    The graph holds what GraphSHAP-IQ and the graph imputer need to know: its edges, its number of
    nodes, the features of its nodes, and the radius of the receptive fields, the number of hops a
    node's embedding depends on. Like :class:`~shapiq.tree.TreeModel` for trees, the converters of
    :mod:`shapiq.graph.conversion` build it from the objects of graph libraries. Unlike a tree, it
    does not describe the model's computation: the model keeps computing its outputs.

    Attributes:
        edge_index: The edges as an integer array of shape ``(2, n_edges)``, from source to target
            nodes. Their direction is ignored for the receptive fields.
        n_hops: The radius of the receptive fields, e.g. the number of message passing layers.
        x: The node features of shape ``(n_nodes, n_features)``, or ``None`` for a graph that is
            only a structure.
        n_nodes: The number of nodes. Defaults to the number of rows of ``x``, else to one more
            than the largest node of the edges.

    The arrays are copied, so the graph does not change with the arrays it was built from.

    Examples:
        >>> import numpy as np
        >>> graph = GraphModel(edge_index=np.array([[0, 1, 2], [1, 2, 3]]), n_hops=1)
        >>> graph.n_nodes, [field.tolist() for field in graph.receptive_fields]
        (4, [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3]])
    """

    edge_index: np.ndarray
    n_hops: int
    x: np.ndarray | None = None
    n_nodes: int | None = None

    def __post_init__(self) -> None:
        """Check the graph and store its arrays as numpy arrays.

        Raises:
            ValueError: If the edges are not integers of shape ``(2, n_edges)`` or refer to other
                nodes, ``n_hops`` or ``n_nodes`` is negative, or the node features are not of
                shape ``(n_nodes, n_features)``.
        """
        edges = np.array(self.edge_index)
        if edges.ndim != 2 or edges.shape[0] != 2:
            msg = f"edge_index must be of shape (2, n_edges), got {edges.shape}."
            raise ValueError(msg)
        if edges.size and not np.issubdtype(edges.dtype, np.integer):
            msg = f"edge_index must hold integer node indices, got {edges.dtype}."
            raise ValueError(msg)
        edges = edges.astype(np.int64)
        n_hops = self.n_hops
        if isinstance(n_hops, bool) or not isinstance(n_hops, int | np.integer) or n_hops < 0:
            msg = f"n_hops must be a non-negative integer, got {n_hops!r}."
            raise ValueError(msg)
        x = None if self.x is None else np.array(self.x)
        if x is not None and x.ndim != 2:
            msg = f"x must be of shape (n_nodes, n_features), got {x.shape}."
            raise ValueError(msg)
        n_nodes = self.n_nodes
        if n_nodes is None:
            n_nodes = x.shape[0] if x is not None else int(edges.max(initial=-1)) + 1
        if n_nodes < 0:
            msg = f"n_nodes must be non-negative, got {n_nodes}."
            raise ValueError(msg)
        if x is not None and x.shape[0] != n_nodes:
            msg = f"x has {x.shape[0]} rows for {n_nodes} nodes."
            raise ValueError(msg)
        if edges.size and (edges.min() < 0 or edges.max() >= n_nodes):
            msg = f"edge_index refers to nodes outside of 0, ..., {n_nodes - 1}."
            raise ValueError(msg)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "n_hops", int(self.n_hops))
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "n_nodes", int(n_nodes))

    @property
    def n_features(self) -> int:
        """The number of node features.

        Raises:
            ValueError: If the graph has no node features.
        """
        if self.x is None:
            msg = "The graph has no node features (x is None)."
            raise ValueError(msg)
        return self.x.shape[1]

    @cached_property
    def receptive_fields(self) -> tuple[np.ndarray, ...]:
        """The sorted nodes within ``n_hops`` hops of every node, including the node itself.

        The edges are treated as undirected, which contains the receptive field of a message
        passing layer for either direction of message flow.
        """
        n_nodes = self.n_nodes or 0
        identity = sparse.eye_array(n_nodes, dtype=bool, format="csr")
        edges = self.edge_index
        adjacency = sparse.csr_array(
            (np.ones(edges.shape[1], dtype=bool), (edges[0], edges[1])), shape=(n_nodes, n_nodes)
        )
        adjacency = adjacency + adjacency.T + identity
        reach = identity
        for _ in range(self.n_hops):
            reach = reach @ adjacency
        reach = sparse.csr_array(reach)
        reach.sort_indices()
        return tuple(
            reach.indices[reach.indptr[node] : reach.indptr[node + 1]].astype(np.int64)
            for node in range(n_nodes)
        )

    @property
    def max_receptive_field_size(self) -> int:
        """The number of nodes of the largest receptive field."""
        return max((len(field) for field in self.receptive_fields), default=0)
