"""GraphSHAP-IQ: exact Shapley interactions of the nodes of graph neural networks.

The embedding of a node after ``L`` message passing layers depends only on the nodes within ``L``
hops, its receptive field. :class:`GraphSHAPIQ` exploits this for any game whose players are the
nodes of a graph: it evaluates the game only on subsets of the receptive fields and computes the
exact interactions from their Möbius coefficients :cite:p:`Muschalik.2025`.

- :class:`GraphExplainer` explains the graph-level output of a graph neural network.
- :class:`GraphGame` is the game of a graph neural network's output for one graph, with the node
  features of absent nodes replaced by a baseline.
- :class:`GraphSHAPIQ` computes the interactions of a game that is local on a graph.

:class:`GraphGame` and :class:`GraphExplainer` require ``torch`` and ``torch_geometric``, which
the ``graph`` extra installs: ``pip install 'shapiq[graph]'``. :class:`GraphSHAPIQ` does not.
"""

from .explainer import GraphExplainer  # noqa: F401  (documented in shapiq.explainer)
from .game import GraphBaseline, GraphGame
from .graphshapiq import GraphSHAPIQ, GraphSHAPIQIndices, receptive_fields

__all__ = [
    "GraphBaseline",
    "GraphGame",
    "GraphSHAPIQ",
    "GraphSHAPIQIndices",
    "receptive_fields",
]
