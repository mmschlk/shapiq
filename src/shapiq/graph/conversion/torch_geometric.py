"""The torch models of graphs as functions of node features (imports torch)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from . import GraphPredictFunction

if TYPE_CHECKING:
    from shapiq.graph.base import GraphModel

__all__ = ["TorchGeometricPredictFunction"]


class TorchGeometricPredictFunction(GraphPredictFunction):
    """A ``torch`` model of graphs called on batches of masked copies of one graph.

    The copies of a call are evaluated in forward passes of ``batch_size`` graphs, as with a
    :class:`torch_geometric.loader.DataLoader`, and every pass is padded to ``batch_size`` graphs, so
    the output for a copy does not depend on the other copies of the call. The model is set to
    evaluation mode and runs without gradients on the device of its parameters.

    Attributes:
        model: The model, called as ``model(x=x, edge_index=edge_index, batch=batch)``.
        batch_size: The number of graphs per forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        graph: GraphModel,
        *,
        class_index: int | None = None,
        batch_size: int = 128,
    ) -> None:
        """Prepare the batches of copies of the graph.

        Raises:
            ValueError: If ``batch_size`` is not positive or the model does not return one output
                per graph.
        """
        if batch_size < 1:
            msg = f"batch_size must be positive, got {batch_size}."
            raise ValueError(msg)
        self.model = model.eval()
        self.batch_size = batch_size
        parameter = next(model.parameters(), None)
        self._device = parameter.device if parameter is not None else torch.device("cpu")
        self._dtype = parameter.dtype if parameter is not None else torch.float32
        n_nodes = graph.n_nodes or 0
        offsets = torch.arange(batch_size, device=self._device) * n_nodes
        edges = torch.as_tensor(graph.edge_index, device=self._device)
        self._edge_index = (edges[:, None, :] + offsets[None, :, None]).reshape(2, -1)
        self._batch = torch.arange(batch_size, device=self._device).repeat_interleave(n_nodes)
        super().__init__(graph, class_index)

    def outputs(self, node_features: np.ndarray) -> np.ndarray:
        """Return the model outputs of shape ``(n_graphs, n_outputs)`` for the node features."""
        n_features = node_features.shape[2]
        outputs = []
        for start in range(0, len(node_features), self.batch_size):
            chunk = np.asarray(node_features[start : start + self.batch_size], dtype=float)
            padding = np.repeat(chunk[-1:], self.batch_size - len(chunk), axis=0)
            x = torch.as_tensor(np.concatenate([chunk, padding]), dtype=self._dtype)
            with torch.no_grad():
                output = self.model(
                    x=x.to(self._device).reshape(-1, n_features),
                    edge_index=self._edge_index,
                    batch=self._batch,
                )
            if output.shape[0] != self.batch_size or output.ndim > 2:
                msg = (
                    f"The model returned an output of shape {tuple(output.shape)} for a batch of "
                    f"{self.batch_size} graphs; it must return one output per graph, e.g. by "
                    "pooling the node embeddings with the batch vector."
                )
                raise ValueError(msg)
            outputs.append(output.reshape(self.batch_size, -1)[: len(chunk)])
        return torch.cat(outputs).detach().cpu().double().numpy()
