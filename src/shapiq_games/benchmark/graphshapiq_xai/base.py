"""This module contains the base GraphSHAP-IQ xai game."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data
    from torch_geometric.nn.models import GAT, GCN, GIN

from shapiq.game import Game


class GraphGame(Game):
    """A GraphSHAP-IQ explanation game for graph networks.

    The game is based on the GraphSHAP-IQ algorithm and is used to explain the predictions of graph
    networks. GraphSHAP-IQ is used to compute Shapley interaction values for graph networks.
    """

    def __init__(
        self,
        model: GCN | GIN | GAT,
        x_graph: Data,
        *,
        class_id: int | None = None,
        baseline_strategy: str | None = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        """Docstring."""
        self.model = model
        self.model.eval()
        self.x_graph = x_graph.clone()
        self.output_dim = 1
        self.edge_index = self.x_graph.edge_index.detach().numpy()

        if baseline_strategy is None:
            warnings.warn(
                "Baseline is not provided, baseline will be initialized as zero...", stacklevel=2
            )
            self.baseline = torch.zeros(x_graph.num_node_features, dtype=torch.float32)
        else:
            self.baseline = self.calculate_baseline(baseline_strategy)

        if class_id is None:  # explaining the predicted class
            # eval model and find the prediction index
            model_output = self.model(
                x=self.x_graph.x, edge_index=self.x_graph.edge_index, batch=self.x_graph.batch
            )
            self.y_index = int(np.argmax(model_output.detach().numpy(), axis=1)[0])
        else:
            self.y_index = int(class_id)

        if normalize:
            # Compute emptyset prediction
            normalization_value = float(self.value_function(np.zeros(len(x_graph.x))))
            # call the super constructor
            super().__init__(
                n_players=len(x_graph.x),
                normalize=normalize,
                normalization_value=normalization_value,
                verbose=verbose,
            )
        else:
            # call the super constructor
            super().__init__(n_players=len(x_graph.x), normalize=normalize)
        self._grand_coalition_set = set(range(self.n_players))

    def calculate_baseline(self, strategy: str) -> torch.tensor:
        """Returns Tensor for replacing node features, depending on the chosen strategy."""
        x = self.x_graph.clone().x
        match strategy:
            case "average":
                return x.mean(dim=0, dtype=torch.float)
            case "min":
                return torch.amin(x, dim=0).to(torch.float)
            case "max":
                return torch.amax(x, dim=0).to(torch.float)
            case "zeros":
                return torch.zeros(self.x_graph.num_node_features, dtype=torch.float32)
            case _:
                warnings.warn(
                    "Unknown baseline strategy, baseline will be initialized as zero...",
                    stacklevel=2,
                )
                return torch.zeros(self.x_graph.num_node_features, dtype=torch.float32)

    def mask_input(self, coalition: np.ndarray) -> Data:
        """The masking procedure for feature-removal. Masks all feature values of masked nodes.

        Args:
            coalition: A binary numpy array containing the masking.

        Returns: The masked x_graph for the coalition as a graph tensor.
        """
        x_masked = self.x_graph.clone()
        x_masked.x = x_masked.x * torch.tensor(
            coalition.reshape((-1, 1)), dtype=torch.float32
        ) + self.baseline * torch.tensor((1 - coalition).reshape((-1, 1)), dtype=torch.float32)
        return x_masked

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Docstring."""
        # TO DO
        return coalitions
