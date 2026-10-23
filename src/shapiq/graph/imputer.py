"""The baseline imputer of graphs: absent nodes take the baseline node features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.imputer import BaselineImputer

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues, Model

    from .base import GraphModel

__all__ = ["GraphBaselineImputer"]


class GraphBaselineImputer(BaselineImputer):
    """The :class:`~shapiq.imputer.BaselineImputer` of a graph, whose players are its nodes.

    Absent nodes keep their edges, but their node features are replaced by the baseline
    :cite:p:`Muschalik.2025`. As for the :class:`~shapiq.imputer.BaselineImputer`, the baseline is
    the mean (the mode for categorical features) of the background node features ``data``, or
    ``data`` itself if it is a single row. The model is called on the node features of the imputed
    copies of the graph, of shape ``(n_coalitions, n_nodes, n_features)``, e.g. a function of
    :func:`~shapiq.graph.conversion.get_graph_predict_function`.

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
            model: A function of node features of shape ``(n_coalitions, n_nodes, n_features)``.
            data: The background node features of shape ``(n_samples, n_features)``, or the
                baseline node features of shape ``(n_features,)``.
            x: The explained graph, with node features.
            categorical_features: The indices of the categorical node features. Defaults to
                ``None``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            random_state: The random state of the imputer. Defaults to ``None``.
        """
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
            ValueError: If the graph has no node features or other ones than the background data.
        """
        if x.x is None or x.x.shape[1] != self.n_features:
            n_features = None if x.x is None else x.x.shape[1]
            msg = (
                f"The graph needs node features of the {self.n_features} features of the "
                f"background data, got {n_features}."
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
        baseline = self.baseline_values.astype(self.x.dtype)
        node_features = np.where(np.asarray(coalitions, dtype=bool)[:, :, None], self.x, baseline)
        return self.predict(node_features)

    def calc_empty_prediction(self) -> float:
        """Run the model on the graph with all nodes absent to get the empty prediction."""
        empty_prediction = float(self.value_function(np.zeros((1, self.n_players), dtype=bool))[0])
        self.empty_prediction = empty_prediction
        if self.normalize:  # reset the normalization value
            self.normalization_value = empty_prediction
        return empty_prediction
