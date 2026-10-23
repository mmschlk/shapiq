"""The game of a graph neural network's prediction for one graph with nodes masked."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from shapiq.game import Game

if TYPE_CHECKING:
    import torch
    from numpy.typing import ArrayLike
    from torch_geometric.data import Data

    from shapiq.typing import CoalitionMatrix, GameValues

__all__ = ["GraphBaseline", "GraphGame", "require_torch_geometric"]

GraphBaseline = Literal["zeros", "mean", "min", "max"]
"""The node features absent nodes take: zeros, or the mean, minimum, or maximum over the nodes."""


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
            "shapiq.graph requires torch and torch_geometric. Install them with: "
            "pip install 'shapiq[graph]'"
        )
        raise ImportError(msg) from error


class GraphGame(Game):
    """The output of a graph neural network for one graph, given a coalition of its nodes.

    The players are the nodes of the graph. The node features of absent nodes are replaced by a
    baseline while the edges are kept :cite:p:`Muschalik.2025`. The value of a coalition is the
    graph-level output of the model for the masked graph: the explained class's logit for a model
    with several outputs, else its single output.

    The game is local on the graph (see :class:`~shapiq.graph.GraphSHAPIQ`) if the model consists
    of ``n_hops`` message passing layers followed by a linear global pooling (e.g. a sum or mean
    over the nodes) and a linear readout. GraphSHAP-IQ then computes its values exactly. A
    softmax or a nonlinear readout after the pooling makes the game non-local, so explain logits.

    The model is called as ``model(x=x, edge_index=edge_index, batch=batch)`` on batches of
    ``batch_size`` masked copies of the graph, as with a
    :class:`torch_geometric.loader.DataLoader`, and must return one output per graph. Every batch
    is padded to ``batch_size`` graphs, so the value of a coalition does not depend on which
    coalitions are evaluated with it. The model is set to evaluation mode.

    Attributes:
        model: The graph neural network.
        graph: The explained graph.
        class_index: The explained output for models with several outputs, ``None`` for models
            with one output.
        n_hops: The number of message passing layers of the model.
        edge_index: The edges of the graph as an integer array of shape ``(2, n_edges)``.
        baseline: The node features of absent nodes, of shape ``(n_node_features,)``.
        batch_size: The number of masked graphs per forward pass.
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
        >>> game.n_players, game.n_hops
        (4, 1)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        graph: Data,
        *,
        class_index: int | None = None,
        n_hops: int | None = None,
        baseline: GraphBaseline | float | ArrayLike = "zeros",
        batch_size: int = 128,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the game.

        Args:
            model: A graph neural network with a graph-level output.
            graph: The explained graph with node features ``graph.x`` and edges
                ``graph.edge_index``. Other attributes of the graph are not passed to the model.
            class_index: The explained output of a model with several outputs. Defaults to
                ``None``, which means output ``1`` for models with several outputs (the convention
                of the shapiq explainers) and the output of models with one output.
            n_hops: The number of message passing layers of the model. Defaults to ``None``,
                which takes ``model.num_layers`` (set by the models of ``torch_geometric.nn``).
            baseline: The node features of absent nodes: ``"zeros"`` (default), ``"mean"``,
                ``"min"``, or ``"max"`` of the node features over the nodes of the graph, a value
                for every feature, or one value per feature.
            batch_size: The number of masked graphs per forward pass. Defaults to ``128``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ImportError: If torch or torch_geometric is not installed.
            ValueError: If the graph has no node features, ``n_hops`` is not given and the model
                has no ``num_layers``, the baseline does not fit the node features, the class
                index is out of range, or the model does not return one output per graph.
        """
        require_torch_geometric()
        import torch

        if graph.x is None or graph.edge_index is None:
            msg = "The graph needs node features (graph.x) and edges (graph.edge_index)."
            raise ValueError(msg)
        if n_hops is None:
            n_hops = getattr(model, "num_layers", None)
        if isinstance(n_hops, bool) or not isinstance(n_hops, int) or n_hops < 0:
            msg = (
                "Pass n_hops, the number of message passing layers of the model: it has no "
                f"non-negative integer num_layers attribute, got n_hops={n_hops!r}."
            )
            raise ValueError(msg)
        if batch_size < 1:
            msg = f"batch_size must be positive, got {batch_size}."
            raise ValueError(msg)

        self.model = model.eval()
        self.graph = graph
        self.n_hops = n_hops
        self.batch_size = batch_size
        self.edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)
        self._x: torch.Tensor = graph.x.detach()
        n_nodes = self._x.shape[0]
        self.baseline: torch.Tensor = _resolve_baseline(baseline, self._x)

        device = self._x.device
        offsets = torch.arange(batch_size, device=device) * n_nodes
        edge_index = graph.edge_index.detach().to(device)
        self._batch_edge_index = (edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)
        self._batch = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)

        n_outputs = self._forward(np.ones((1, n_nodes), dtype=bool)).shape[1]
        self.class_index = _resolve_class_index(class_index, n_outputs)
        self.empty_value = float(self.value_function(np.zeros((1, n_nodes), dtype=bool))[0])
        self.original_model_output = float(
            self.value_function(np.ones((1, n_nodes), dtype=bool))[0]
        )
        super().__init__(
            n_nodes,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def _forward(self, coalitions: np.ndarray) -> torch.Tensor:
        """Return the model outputs of shape ``(n_coalitions, n_outputs)`` for the masked graphs."""
        import torch

        n_features = self._x.shape[1]
        outputs = []
        for start in range(0, len(coalitions), self.batch_size):
            chunk = coalitions[start : start + self.batch_size]
            padded = np.concatenate([chunk, np.repeat(chunk[-1:], self.batch_size - len(chunk), 0)])
            present = torch.as_tensor(padded, device=self._x.device)[..., None]
            x = torch.where(present, self._x, self.baseline).reshape(-1, n_features)
            with torch.no_grad():
                output = self.model(x=x, edge_index=self._batch_edge_index, batch=self._batch)
            if output.shape[0] != self.batch_size or output.ndim > 2:
                msg = (
                    f"The model returned an output of shape {tuple(output.shape)} for a batch of "
                    f"{self.batch_size} graphs; it must return one output per graph, e.g. by "
                    "pooling the node embeddings with the batch vector."
                )
                raise ValueError(msg)
            outputs.append(output.reshape(self.batch_size, -1)[: len(chunk)])
        return torch.cat(outputs)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the model's output for the graph with the absent nodes masked.

        Args:
            coalitions: The coalitions as a boolean matrix of shape ``(n_coalitions, n_nodes)``.

        Returns:
            The explained output for every coalition.
        """
        coalitions = np.asarray(coalitions, dtype=bool).reshape(-1, self._x.shape[0])
        output = self._forward(coalitions)[:, 0 if self.class_index is None else self.class_index]
        return output.detach().cpu().double().numpy()


def _resolve_baseline(baseline: GraphBaseline | float | ArrayLike, x: torch.Tensor) -> torch.Tensor:
    """Return the baseline node features as a tensor of shape ``(n_node_features,)``."""
    import torch

    if isinstance(baseline, str):
        match baseline:
            case "zeros":
                return torch.zeros_like(x[0])
            case "mean":
                return x.mean(dim=0)
            case "min":
                return x.amin(dim=0)
            case "max":
                return x.amax(dim=0)
        msg = f"baseline must be 'zeros', 'mean', 'min', 'max', or numeric, got {baseline!r}."
        raise ValueError(msg)
    values = torch.as_tensor(np.asarray(baseline, dtype=float), dtype=x.dtype, device=x.device)
    if values.ndim == 0:
        return values.expand(x.shape[1]).clone()
    if values.shape != (x.shape[1],):
        msg = f"baseline must have one value per node feature ({x.shape[1]}), got {values.shape}."
        raise ValueError(msg)
    return values


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
