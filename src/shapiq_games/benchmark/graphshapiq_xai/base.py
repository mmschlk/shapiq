"""This module contains the GraphGame for GraphSHAP-IQ."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data

from shapiq.game import Game


class GraphGame(Game):
    """A GraphSHAP-IQ explanation game for graph networks.

    The game is based on the GraphSHAP-IQ algorithm and is used to explain the predictions of graph
    networks. GraphSHAP-IQ is used to compute Shapley interaction values for graph networks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x_graph: Data,
        *,
        class_index: int | None = None,
        baseline_strategy: str | None = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize the GraphGame.

        Args:
            model: A GNN model (GCN, GIN, or GAT) used to compute predictions.
            x_graph: The input graph as a torch_geometric Data object.
            class_index: The target class index for classification. If None, the predicted class is
                used.
            baseline_strategy: Strategy for replacing masked node features. One of "average",
                "min", "max", or "zeros". If None, defaults to zeros with a warning.
            normalize: Whether to normalize the game by the empty coalition prediction.
            verbose: Whether to show progress bars during evaluation.
        """
        self.model = model
        self.model.eval()
        self.x_graph = x_graph.clone()

        if baseline_strategy is None:
            warnings.warn(
                "Baseline is not provided, baseline will be initialized as zero...", stacklevel=2
            )
            self.baseline = torch.zeros(x_graph.num_node_features, dtype=torch.float32)
        else:
            self.baseline = self.calculate_baseline(baseline_strategy)

        if class_index is None:
            model_output = self.model(
                x=self.x_graph.x, edge_index=self.x_graph.edge_index, batch=self.x_graph.batch
            )
            self.y_index = int(np.argmax(model_output.detach().numpy(), axis=1)[0])
        else:
            self.y_index = int(class_index)

        if normalize:
            normalization_value = float(self.value_function(np.zeros(len(x_graph.x))))
            super().__init__(
                n_players=len(x_graph.x),
                normalize=normalize,
                normalization_value=normalization_value,
                verbose=verbose,
            )
        else:
            super().__init__(n_players=len(x_graph.x), normalize=normalize)
        self._grand_coalition_set = set(range(self.n_players))

    def calculate_baseline(self, strategy: str) -> torch.Tensor:
        """Returns a tensor for replacing node features depending on the chosen strategy."""

        # No deep copy here, since the x_graph is not modified
        x = self.x_graph.x
        match strategy:
            case "average":
                return x.mean(dim=0)
            case "min":
                return torch.amin(x, dim=0)
            case "max":
                return torch.amax(x, dim=0)
            case "zeros":

                # Device is needed for the zeros tensor -> possible that the device is not the same as the model
                return torch.zeros(self.x_graph.num_node_features, dtype=torch.float32,
                                   device=x.device)
            case _:
                warnings.warn(
                    "Unknown baseline strategy, baseline will be initialized as zero...",
                    stacklevel=2,
                )
                return torch.zeros(self.x_graph.num_node_features, dtype=torch.float32,
                                   device=x.device)

    def mask_input(self, coalition: np.ndarray) -> Data:
        """Mask inactive node features with the baseline.

        Args:
            coalition: A binary numpy array where 1 = active node, 0 = inactive.

        Returns:
            A cloned graph with inactive nodes replaced by the baseline features.
        """

        # Convert coalition to boolean tensor on the same device as the model
        coalition_tensor = torch.tensor(coalition, dtype=torch.bool, device=self.x_graph.x.device)
        x_masked = self.x_graph.clone()

        # Reshape the baseline to match the number of features in the graph
        baseline_reshaped = self.baseline.reshape(1, -1)
        x_masked.x[~coalition_tensor] = baseline_reshaped

        return x_masked

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the GNN for each coalition by masking inactive node features.

        Args:
            coalitions: Binary matrix of shape (n_coalitions, n_nodes). A 1D array of shape
                (n_nodes,) is also accepted and reshaped automatically.

        Returns:
            Array of shape (n_coalitions,) containing one model prediction per coalition.
        """
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        coalition_values = []

        for coalition in coalitions:
            masked_graph = self.mask_input(coalition)

            with torch.no_grad():
                model_output = self.model(
                    x=masked_graph.x,
                    edge_index=masked_graph.edge_index,
                    batch=getattr(masked_graph, "batch", None),
                )

            if model_output.ndim > 1:
                coalition_value = model_output[0, self.y_index]
            else:
                coalition_value = model_output.squeeze()

            coalition_values.append(float(coalition_value))

        return np.asarray(coalition_values)