"""The baseline imputer of graphs: absent nodes take the baseline node features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.imputer import BaselineImputer
from shapiq.utils.modules import safe_isinstance

from .base import GraphModel
from .conversion import get_graph_predict_function

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues, Model

__all__ = ["GraphBaselineImputer"]

_MAX_VALUES = 2**22
"""The most node feature values imputed at once (32 MB of floats)."""


class GraphBaselineImputer(BaselineImputer):
    """The :class:`~shapiq.imputer.BaselineImputer` of a graph, whose players are its nodes.

    Absent nodes keep their edges, but their node features are replaced by the baseline
    :cite:p:`Muschalik.2025`. As for the :class:`~shapiq.imputer.BaselineImputer`, the baseline is
    the mean (the mode for categorical features, by default all integer node features) of the
    background node features ``data``, or ``data`` itself if it is a single row. The model is
    called on the node features of the imputed copies of the graph, of shape
    ``(n_coalitions, n_nodes, n_features)``, in chunks of a few million node feature values. A
    ``torch`` model is turned into such a function with
    :func:`~shapiq.graph.get_graph_predict_function`.

    The imputer is bound to the structure of its graph: it can be refitted to the node features of
    another graph with the same nodes and edges only.

    Attributes:
        graph: The explained graph.

    Examples:
        >>> import numpy as np
        >>> from shapiq.graph import GraphModel
        >>> graph = GraphModel(edge_index=np.array([[0, 1], [1, 2]]), n_hops=1, x=np.ones((3, 2)))
        >>> def model(node_features):  # the sum over the nodes of the first node feature
        ...     return node_features[:, :, 0].sum(axis=1)
        >>> imputer = GraphBaselineImputer(model, data=np.zeros(2), x=graph)
        >>> imputer.n_players, imputer(np.array([[True, False, True]]))
        (3, array([2.]))
    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        x: GraphModel,
        *,
        categorical_features: list[int] | None = None,
        normalize: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the imputer.

        Args:
            model: A function of node features of shape ``(n_coalitions, n_nodes, n_features)``,
                or a ``torch`` model of graphs (see
                :func:`~shapiq.graph.get_graph_predict_function`).
            data: The background node features of shape ``(n_samples, n_features)``, or the
                baseline node features of shape ``(n_features,)``.
            x: The explained graph, with node features.
            categorical_features: The indices of the categorical node features. Defaults to
                ``None``, which means all node features for integer node features and none
                otherwise.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            random_state: The random state of the imputer. Defaults to ``None``.
        """
        if safe_isinstance(model, "torch.nn.Module"):
            model = get_graph_predict_function(model, x)
        if categorical_features is None and x.x is not None and x.x.dtype.kind in "biu":
            categorical_features = list(range(x.x.shape[1]))
        super().__init__(
            model,
            data,
            x,  # ty: ignore[invalid-argument-type]
            categorical_features=categorical_features,
            normalize=normalize,
            random_state=random_state,
        )

    def fit(self, x: GraphModel) -> GraphBaselineImputer:  # type: ignore[override]
        """Set the explained graph.

        Raises:
            ValueError: If the graph has no node features or other ones than the background data,
                or other nodes or edges than the graph of the imputer or of its model.
        """
        if x.x is None or x.x.shape[1] != self.n_features:
            n_features = None if x.x is None else x.x.shape[1]
            msg = (
                f"The graph needs node features of the {self.n_features} features of the "
                f"background data, got {n_features}."
            )
            raise ValueError(msg)
        for graph in (getattr(self, "graph", None), getattr(self.model, "graph", None)):
            if isinstance(graph, GraphModel) and not _same_structure(graph, x):
                msg = (
                    "The graph has other nodes or edges than the graph the imputer or its model "
                    "was built for: build a new imputer for it."
                )
                raise ValueError(msg)
        self.graph = x
        self._x = x.x.copy()
        return self

    def _n_players(self) -> int:
        """The nodes are the players."""
        return self.graph.n_nodes or 0

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the model's output for the graph with the absent nodes' features imputed.

        Args:
            coalitions: The coalitions as a boolean matrix of shape ``(n_coalitions, n_nodes)``.

        Returns:
            The model's output for every coalition.
        """
        coalitions = np.asarray(coalitions, dtype=bool)
        if not len(coalitions):
            return np.zeros(0)
        x = self.x
        baseline = self.baseline_values.astype(x.dtype)
        rows = max(1, _MAX_VALUES // x.size)
        return np.concatenate(
            [
                np.asarray(self.predict(np.where(chunk[:, :, None], x, baseline)), dtype=float)
                for chunk in np.split(coalitions, range(rows, len(coalitions), rows))
            ]
        )

    def calc_empty_prediction(self) -> float:
        """Run the model on the graph with all nodes absent to get the empty prediction."""
        empty_prediction = float(self.value_function(np.zeros((1, self.n_players), dtype=bool))[0])
        self.empty_prediction = empty_prediction
        if self.normalize:  # reset the normalization value
            self.normalization_value = empty_prediction
        return empty_prediction


def _same_structure(graph: GraphModel, other: GraphModel) -> bool:
    """Return whether two graphs have the same nodes and edges."""
    return graph.n_nodes == other.n_nodes and np.array_equal(graph.edge_index, other.edge_index)
