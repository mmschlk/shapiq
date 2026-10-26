"""GraphSHAP-IQ: exact Shapley interactions of the nodes of graph neural networks.

The embedding of a node after ``L`` message passing layers depends only on the nodes within ``L``
hops, its receptive field. :class:`GraphSHAPIQ` exploits this for any game whose players are the
nodes of a graph: it evaluates the game only on subsets of the receptive fields and computes the
exact interactions from their Möbius coefficients :cite:p:`Muschalik.2025`.

- :class:`GraphModel` is the internal representation of a graph: its edges, nodes, node features,
  and the radius of the receptive fields. :func:`convert_graph` builds it from a graph of
  ``torch_geometric``.
- :class:`GraphBaselineImputer` is the :class:`~shapiq.imputer.BaselineImputer` of a graph: its
  players are the nodes, and absent nodes take the baseline node features. It calls the model as
  a function of node features, which :func:`get_graph_predict_function` makes of a ``torch``
  model.
- :class:`GraphSHAPIQ` computes the interactions of a game that is local on a graph.
- :class:`~shapiq.explainer.GraphExplainer` explains the graph-level output of a graph neural
  network with them.

Graphs and models of ``torch_geometric`` require ``torch`` and ``torch_geometric``, which the
``graph`` extra installs: ``pip install 'shapiq[graph]'``. The rest needs neither.
"""

from .base import GraphModel
from .conversion import GraphPredictFunction, convert_graph, get_graph_predict_function
from .explainer import GraphExplainer  # noqa: F401  (documented in shapiq.explainer)
from .graphshapiq import GraphSHAPIQ, GraphSHAPIQIndices
from .imputer import GraphBaselineImputer

__all__ = [
    "GraphBaselineImputer",
    "GraphModel",
    "GraphPredictFunction",
    "GraphSHAPIQ",
    "GraphSHAPIQIndices",
    "convert_graph",
    "get_graph_predict_function",
]
