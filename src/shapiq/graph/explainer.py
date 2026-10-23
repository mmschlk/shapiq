"""The GraphExplainer: exact Shapley interactions of the nodes of graph neural networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np
from tqdm.auto import tqdm

from shapiq.explainer.base import Explainer

from .base import GraphModel
from .conversion import convert_graph, get_graph_predict_function, resolve_n_hops
from .graphshapiq import GraphSHAPIQ, GraphSHAPIQIndices
from .imputer import GraphBaselineImputer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_geometric.data import Data

    from shapiq.interaction_values import InteractionValues

__all__ = ["GraphExplainer"]


class GraphExplainer(Explainer):
    """The GraphExplainer explains graph neural networks with GraphSHAP-IQ.

    The players are the nodes of the explained graph. For every graph, the explainer converts it
    into a :class:`~shapiq.graph.GraphModel`, imputes absent nodes with the
    :class:`~shapiq.graph.GraphBaselineImputer`, and computes the interactions of the resulting game
    with :class:`~shapiq.graph.GraphSHAPIQ` :cite:p:`Muschalik.2025`. GraphSHAP-IQ evaluates the
    model only on the subsets of the receptive fields of the nodes, which makes the computation
    feasible for graphs far beyond the reach of evaluating all coalitions.

    The values are exact for models of message passing layers followed by a linear global pooling
    (e.g. a sum or mean over the nodes) and a linear readout, explained on their logits. With
    ``max_interaction_size``, GraphSHAP-IQ evaluates fewer coalitions and approximates the values;
    the returned values are then flagged as ``estimated``.

    Examples:
        >>> import torch
        >>> from torch_geometric.data import Data
        >>> from torch_geometric.nn import GCNConv, global_mean_pool
        >>> class GCN(torch.nn.Module):
        ...     num_layers = 1
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv, self.readout = GCNConv(3, 8), torch.nn.Linear(8, 1)
        ...     def forward(self, x, edge_index, batch):
        ...         return self.readout(global_mean_pool(self.conv(x, edge_index).relu(), batch))
        >>> graph = Data(x=torch.rand(4, 3), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        >>> explainer = GraphExplainer(GCN(), index="k-SII", max_order=2)
        >>> interaction_values = explainer.explain(graph)
        >>> interaction_values.estimated
        False
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: np.ndarray | None = None,
        *,
        index: GraphSHAPIQIndices = "SV",
        max_order: int = 1,
        class_index: int | None = None,
        n_hops: int | None = None,
        max_interaction_size: int | None = None,
        efficiency_routine: bool = True,
        batch_size: int = 128,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the GraphExplainer.

        Args:
            model: A graph neural network with a graph-level output, called as
                ``model(x=x, edge_index=edge_index, batch=batch)``, or a function of node features
                of shape ``(n_graphs, n_nodes, n_features)`` for graphs given as
                :class:`~shapiq.graph.GraphModel` (see
                :func:`~shapiq.graph.get_graph_predict_function`).
            data: The background node features of shape ``(n_samples, n_features)``, whose mean is
                the baseline of absent nodes, or the baseline node features of shape
                ``(n_features,)`` (see :class:`~shapiq.graph.GraphBaselineImputer`). Defaults to
                ``None``, which means zeros.
            index: The interaction index, ``"Moebius"`` or an index of
                :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter`. Defaults to
                ``"SV"``.
            max_order: The highest interaction order. Defaults to ``1``.
            class_index: The explained output of a model with several outputs. Defaults to
                ``None``, which means output ``1`` for models with several outputs and the output
                of models with one output.
            n_hops: The number of message passing layers of the model. Defaults to ``None``,
                which takes ``model.num_layers`` (or ``n_hops`` of a
                :class:`~shapiq.graph.GraphModel`).
            max_interaction_size: The largest coalitions whose Möbius coefficients GraphSHAP-IQ
                computes. Defaults to ``None``, the largest receptive field, which computes exact
                values.
            efficiency_routine: Whether GraphSHAP-IQ keeps approximated values efficient. Defaults
                to ``True``.
            batch_size: The number of graphs per forward pass of a ``torch`` model. Defaults to
                ``128``.
            **kwargs: Additional keyword arguments are ignored.

        Raises:
            ValueError: If the index is not supported by GraphSHAP-IQ.
        """
        if index not in get_args(GraphSHAPIQIndices):
            msg = f"GraphExplainer supports the indices {get_args(GraphSHAPIQIndices)}, got {index!r}."
            raise ValueError(msg)
        super().__init__(model, class_index=class_index, index=index, max_order=max_order)
        self._background = None if data is None else np.asarray(data)
        self.class_index = class_index
        self.n_hops = n_hops
        self.max_interaction_size = max_interaction_size
        self.efficiency_routine = efficiency_routine
        self.batch_size = batch_size

    def explain_function(  # type: ignore[override]
        self,
        x: Data | GraphModel,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Compute the interaction values of the nodes of a graph.

        Args:
            x: The explained graph, a ``torch_geometric.data.Data`` graph or a
                :class:`~shapiq.graph.GraphModel` with node features.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            The interaction values, with the output for the graph without nodes as baseline.
        """
        if isinstance(x, GraphModel):
            graph = convert_graph(x, n_hops=self.n_hops)
        else:
            graph = convert_graph(x, n_hops=resolve_n_hops(self.model, self.n_hops))
        predict = get_graph_predict_function(
            self.model, graph, class_index=self.class_index, batch_size=self.batch_size
        )
        background = np.zeros(graph.n_features) if self._background is None else self._background
        imputer = GraphBaselineImputer(predict, background, x=graph, normalize=False)
        graphshapiq = GraphSHAPIQ(
            imputer,
            graph,
            max_interaction_size=self.max_interaction_size,
            efficiency_routine=self.efficiency_routine,
        )
        return graphshapiq(self._index, self._max_order)  # ty: ignore[invalid-argument-type]

    def explain_X(  # type: ignore[override]
        self,
        X: Sequence[Data | GraphModel],
        *,
        n_jobs: int | None = None,
        random_state: int | None = None,  # noqa: ARG002
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[InteractionValues]:
        """Explain several graphs.

        Args:
            X: The explained graphs.
            n_jobs: Number of jobs for ``joblib.Parallel``. Defaults to ``None``, which will use no
                parallelization. If set to ``-1``, all available cores will be used.
            random_state: Ignored, as the explanations are deterministic.
            verbose: Whether to show a progress bar. Defaults to ``False``.
            **kwargs: Additional keyword arguments passed to :meth:`explain_function`.

        Returns:
            The interaction values of every graph.
        """
        if n_jobs:
            import joblib

            parallel = joblib.Parallel(n_jobs=n_jobs)
            delayed = joblib.delayed(self.explain_function)
            return list(parallel(delayed(graph, **kwargs) for graph in X))
        graphs = tqdm(X, desc="Explaining") if verbose else X
        return [self.explain_function(graph, **kwargs) for graph in graphs]
