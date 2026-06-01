"""This module contains the GraphGame for GraphSHAP-IQ."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

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
        task: Literal["classification", "regression"] = "classification",
        class_index: int | None = None,
        output_dim: int = 1,
        baseline_strategy: str | float | None = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize the GraphGame.

        Args:
            model: A GNN model used to compute predictions.
            x_graph: The input graph as a torch_geometric Data object.
            task: Whether the model performs "classification" or "regression".
            class_index: Target class index for classification. If None, the predicted class is
                used. Must not be set when task is "regression".
            baseline_strategy: Strategy for replacing masked node features. Either a string
                "average" to compute the baseline from the graph data, or an explicit float of shape
                used directly as the baseline. If ``None``, defaults
                to zeros with a warning.
            normalize: Whether to normalize the game by the empty coalition prediction.
                If True, the game values are centered such that the empty coalition has a value of 0.
            verbose: Whether to show progress bars during evaluation.
            output_dim: The output dimension of the graphgame.
        """
        self._normalize = normalize
        self.x_graph = x_graph.clone()

        if task not in ("classification", "regression"):
            msg = f"task must be 'classification' or 'regression', got {task!r}"
            raise ValueError(msg)
        if task == "regression" and class_index is not None:
            msg = "class_index cannot be set for regression tasks."
            raise ValueError(msg)
        if self.x_graph.x is None:
            msg = "x_graph must have node features (x_graph.x must not be None)."
            raise ValueError(msg)

        self.task = task
        self.model = model
        self.model.eval()

        self.edge_index = self.x_graph.edge_index.detach().numpy()  # pyright: ignore[reportOptionalMemberAccess]
        self.max_neighborhood_size = model.num_layers
        self.output_dim = output_dim
        self.grand_coalition_set = set(range(self.x_graph.x.shape[0]))

        # Initialize baseline for masking
        if baseline_strategy is None:
            warnings.warn(
                "Baseline is not provided, baseline will be initialized as zero...",
                stacklevel=2,
            )
            self.baseline = torch.zeros(
                x_graph.num_node_features,
                dtype=torch.float32,
                device=self.x_graph.x.device,  # Ensure baseline is on the same device as the graph
            )
        elif isinstance(baseline_strategy, float):
            expected_shape = (x_graph.num_node_features,)
            self.baseline = torch.full(expected_shape, baseline_strategy, dtype=torch.float32)
        elif baseline_strategy == "average":
            self.baseline = self.x_graph.x.mean(dim=0)
        else:
            err_msg = f"Baseline strategy '{baseline_strategy}' is not supported"
            raise NotImplementedError(err_msg)

        if task == "classification":
            if class_index is None:
                model_output = self.model(
                    x=self.x_graph.x,
                    edge_index=self.x_graph.edge_index,
                    batch=self.x_graph.batch,
                )
                self.y_index: int | None = int(np.argmax(model_output.detach().numpy(), axis=1)[0])
            else:
                self.y_index = int(class_index)
        else:
            self.y_index = None

        n_nodes = self.x_graph.x.size(0)
        if normalize:
            normalization_value = float(self.value_function(np.zeros(n_nodes)))
            super().__init__(
                n_players=n_nodes,
                normalize=normalize,
                normalization_value=normalization_value,
                verbose=verbose,
            )
        else:
            super().__init__(n_players=n_nodes, normalize=normalize, verbose=verbose)

        # Update normalization_value if normalize=True
        if normalize:
            # Compute the value for the empty coalition (all nodes masked)
            empty_coalition_value = self.value_function(np.zeros(len(self.x_graph.x)))
            # Extract the scalar value (assuming empty_coalition_value is a 1D array with one element)
            self.normalization_value = float(empty_coalition_value[0])

    @property
    def normalize(self) -> bool:
        """Override the normalize property to return the actual normalize flag.

        Returns:
            bool: True if the game is normalized, False otherwise.
        """
        return self._normalize

    def mask_input(self, coalition: np.ndarray) -> Data:
        """Mask inactive node features with the baseline.

        Args:
            coalition: A binary numpy array where 1 = active node, 0 = inactive.

        Returns:
            A cloned graph with inactive nodes replaced by the baseline features.
        """
        # Convert coalition to boolean tensor on the same device as the model
        coalition_tensor = torch.tensor(coalition, dtype=torch.bool, device=self.x_graph.x.device)  # pyright: ignore[reportOptionalMemberAccess]
        x_masked = self.x_graph.clone()

        # Reshape the baseline to match the number of features in the graph
        baseline_reshaped = self.baseline.reshape(1, -1)
        x_masked.x[~coalition_tensor] = baseline_reshaped  # pyright: ignore[reportOptionalSubscript]

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

            if self.task == "classification":
                # Output shape: (1, num_classes) — select the target class score.
                coalition_value = model_output[0, self.y_index]
            else:
                coalition_value = model_output.squeeze()

            coalition_values.append(float(coalition_value))

        return np.asarray(coalition_values)
