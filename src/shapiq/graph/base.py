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
    """GraphSHAP-IQ explanation game for graph neural networks.

    The game evaluates a graph neural network on masked versions of a given
    input graph. Each node is treated as one player. For a coalition, nodes
    inside the coalition keep their original node features, while nodes outside
    the coalition are replaced by a baseline feature vector.

    The resulting model output defines the value of the coalition and can be
    used to compute Shapley interaction values for GraphSHAP-IQ.

    Attributes:
        model: The graph neural network used for prediction.
        x_graph: A cloned version of the input graph instance to explain.
        class_index: Output index to explain. If ``None``, the model output is
        expected to be scalar and is used directly.
        edge_index: The graph connectivity in COO format as a NumPy array.
        l_hop_distance: The number of message-passing layers of the GNN.
        n_players: The number of players in the game, equal to the number of
            nodes in ``x_graph``.
        grand_coalition_set: Set containing all node indices of the graph.
        baseline: Feature vector used to replace inactive node features.
        normalization_value: Value of the empty coalition used for
            normalization when ``normalize`` is enabled.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x_graph: Data,
        *,
        class_index: int | None = None,
        baseline_strategy: Literal["zeros", "average", "min", "max"] = "zeros",
        baseline_value: float | torch.Tensor | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the graph explanation game.

        Args:
            model: Graph neural network to explain. The model must define a
                ``num_layers`` attribute indicating the number of message-passing
                layers.
            x_graph: Input graph instance to explain. The graph must contain node
                features in ``x_graph.x`` and an ``edge_index`` attribute.
            class_index: Class index to explain for classification tasks. If
                ``None``, the predicted class of ``model`` on ``x_graph`` is used.
                Must be ``None`` for regression tasks.
            baseline_strategy: Strategy used to compute the baseline feature vector for
                inactive nodes. Supported values are ``"zeros"``, ``"average"``,
                ``"min"``, and ``"max"``.
            baseline_value: Explicit baseline used for inactive node features. If
                provided, this overrides ``baseline_strategy``. A float creates a
                constant baseline vector, while a tensor specifies one baseline value
                per node feature.
            normalize: Whether to normalize the game by subtracting the empty
                coalition value.
            verbose: Whether to print progress information inherited from
                :class:`shapiq.game.Game`.

        Raises:
            AttributeError: If ``model`` does not define ``num_layers``.
            TypeError: If ``model.num_layers`` is not an integer.
            NotImplementedError: If ``baseline_strategy`` is unsupported.
        """
        self._normalize = normalize
        self.x_graph = x_graph.clone()

        if self.x_graph.x is None:
            msg = "x_graph must have node features (x_graph.x must not be None)."
            raise ValueError(msg)
        if not hasattr(model, "num_layers"):
            msg = "The GNN needs a num_layers attribute"
            raise AttributeError(msg)

        self.model = model
        self.model.eval()
        self.edge_index = self.x_graph.edge_index.detach().numpy()
        if not isinstance(model.num_layers, int):
            msg = "model.num_layers must be an int"
            raise TypeError(msg)
        self.l_hop_distance = model.num_layers
        self.n_players = self.x_graph.x.shape[0]
        self.grand_coalition_set = set(range(self.n_players))

        self.baseline = self._calculate_baseline(
            strategy=baseline_strategy,
            value=baseline_value,
        )

        self.class_index = class_index

        if normalize:
            empty_coalition_value = self.value_function(np.zeros(self.n_players))
            normalization_value = float(empty_coalition_value[0])
            super().__init__(
                n_players=self.n_players,
                normalize=normalize,
                normalization_value=normalization_value,
                verbose=verbose,
            )
        else:
            super().__init__(
                n_players=self.n_players,
                normalize=normalize,
                verbose=verbose)

        if normalize:
            self.normalization_value = normalization_value

    from typing import Literal

    def _calculate_baseline(
            self,
            strategy: Literal["zeros", "average", "min", "max"],
            value: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate the baseline feature vector for masked nodes.

        Args:
            strategy: Strategy used to compute the baseline when ``value`` is
                ``None``.
            value: Explicit baseline. If provided, this overrides ``strategy``.
                A float creates a constant baseline vector. A tensor is interpreted
                as a fixed feature-wise baseline.

        Returns:
            Baseline tensor with shape ``(n_features,)``.

        Raises:
            ValueError: If a provided tensor baseline has an invalid shape.
        """
        x = self.x_graph.x

        if value is not None:
            if isinstance(value, torch.Tensor):
                if value.shape != (x.shape[1],):
                    raise ValueError(
                        f"Baseline tensor must have shape ({x.shape[1]},), "
                        f"got {tuple(value.shape)}."
                    )
                return value.to(dtype=torch.float32, device=x.device)

            if isinstance(value, float):
                return torch.full(
                    (x.shape[1],),
                    value,
                    dtype=torch.float32,
                    device=x.device,
                )

            raise TypeError(
                "baseline_value must be a float, torch.Tensor, or None."
            )

        if strategy == "zeros":
            return torch.zeros(x.shape[1], dtype=torch.float32, device=x.device)
        if strategy == "average":
            return x.mean(dim=0)
        if strategy == "min":
            return torch.amin(x, dim=0)
        if strategy == "max":
            return torch.amax(x, dim=0)

        # Should never happen because of the Literal type.
        raise RuntimeError(f"Unexpected baseline strategy: {strategy}")

    @property
    def normalize(self) -> bool:
        """Override the normalize property to return the actual normalize flag."""
        return self._normalize

    def mask_input(self, coalition: np.ndarray) -> Data:
        """Create a masked graph for a coalition.

            Args:
                coalition: Boolean or binary array of shape ``(n_players,)`` indicating
                    which nodes are active.

            Returns:
                A cloned graph where inactive node features are replaced by the
                baseline feature vector.
            """
        coalition_tensor = torch.tensor(
            coalition,
            dtype=torch.bool,
            device=self.x_graph.x.device
        )
        x_masked = self.x_graph.clone()
        baseline_reshaped = self.baseline.reshape(1, -1)
        x_masked.x[~coalition_tensor] = baseline_reshaped
        return x_masked

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate coalition values by running the GNN on masked graphs.

            Args:
                coalitions: Boolean or binary coalition matrix of shape
                    ``(n_coalitions, n_players)``. A one-dimensional array of shape
                    ``(n_players,)`` is interpreted as a single coalition.

            Returns:
                One-dimensional NumPy array with one scalar model output per coalition.
                For classification, the selected class logit is returned. For
                regression, the scalar model output is returned.
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

            if self.class_index is None:
                coalition_value = model_output.squeeze()
            else:
                coalition_value = model_output[0, self.class_index]

            if isinstance(coalition_value, torch.Tensor):
                coalition_values.append(coalition_value.item())
            else:
                coalition_values.append(float(coalition_value))

        return np.array(coalition_values)
