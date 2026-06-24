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
    """A GraphSHAP-IQ explanation game for graph networks."""

    def __init__(
        self,
        model: torch.nn.Module,
        x_graph: Data,
        *,
        task: Literal["classification", "regression"] = "classification",
        class_index: int | None = None,
        baseline_strategy: str | float | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the GraphGame."""
        self._normalize = normalize
        self.x_graph = x_graph.clone()

        # Validierungen
        if task not in ("classification", "regression"):
            msg = f"task must be 'classification' or 'regression', got {task!r}"
            raise ValueError(msg)
        if task == "regression" and class_index is not None:
            msg = "class_index cannot be set for regression tasks."
            raise ValueError(msg)
        if self.x_graph.x is None:
            msg = "x_graph must have node features (x_graph.x must not be None)."
            raise ValueError(msg)
        if not hasattr(model, "num_layers"):
            msg = "The GNN needs a num_layers attribute"
            raise AttributeError(msg)

        self.task = task
        self.model = model
        self.model.eval()
        self.edge_index = self.x_graph.edge_index.detach().numpy()
        if not isinstance(model.num_layers, int):
            msg = "model.num_layers must be an int"
            raise TypeError(msg)
        self.max_neighborhood_size = model.num_layers
        self.n_players = self.x_graph.x.shape[0]  # <-- WICHTIG: n_players als Attribut setzen!
        self.grand_coalition_set = set(range(self.n_players))

        # Baseline initialisieren
        self.baseline = self._calculate_baseline(baseline_strategy)

        # y_index für Klassifizierung setzen
        if task == "classification":
            if class_index is None:
                with torch.no_grad():
                    model_output = self.model(
                        x=self.x_graph.x,
                        edge_index=self.x_graph.edge_index,
                        batch=getattr(self.x_graph, "batch", None),
                    )
                self.y_index = int(model_output.argmax(dim=1)[0].item())
            else:
                self.y_index = int(class_index)
        else:
            self.y_index = None

        # Game-Klasse initialisieren
        if normalize:
            # Berechne den Normalisierungswert (empty coalition)
            empty_coalition_value = self.value_function(np.zeros(self.n_players))
            # Extrahiere den Scalar (value_function gibt jetzt Skalare zurück)
            normalization_value = float(empty_coalition_value[0])
            super().__init__(
                n_players=self.n_players,
                normalize=normalize,
                normalization_value=normalization_value,
                verbose=verbose,
            )
        else:
            super().__init__(n_players=self.n_players, normalize=normalize, verbose=verbose)

        # Aktualisiere normalization_value in der Instanz (falls nötig)
        if normalize:
            self.normalization_value = normalization_value

    def _calculate_baseline(self, strategy: str | float | None) -> torch.Tensor:
        """Berechnet die Baseline für Masking basierend auf der Strategie."""
        if strategy is None:
            warnings.warn(
                "Baseline is not provided, baseline will be initialized as zero...", stacklevel=2
            )
            strategy = "zeros"

        x = self.x_graph.x
        if isinstance(strategy, torch.Tensor):
            if strategy.shape != (x.shape[1],):
                msg = (
                    f"Baseline tensor must have shape ({x.shape[1]},), got {tuple(strategy.shape)}."
                )
                raise ValueError(msg)
            return strategy.to(dtype=torch.float32, device=x.device)
        if isinstance(strategy, float):
            return torch.full((x.shape[1],), strategy, dtype=torch.float32, device=x.device)
        if strategy == "zeros":
            return torch.zeros(x.shape[1], dtype=torch.float32, device=x.device)
        if strategy == "average":
            return x.mean(dim=0)
        if strategy == "min":
            return torch.amin(x, dim=0)
        if strategy == "max":
            return torch.amax(x, dim=0)
        msg = f"Baseline strategy '{strategy}' is not supported."
        raise NotImplementedError(msg)

    @property
    def normalize(self) -> bool:
        """Override the normalize property to return the actual normalize flag."""
        return self._normalize

    def mask_input(self, coalition: np.ndarray) -> Data:
        """Mask inactive node features with the baseline."""
        coalition_tensor = torch.tensor(coalition, dtype=torch.bool, device=self.x_graph.x.device)
        x_masked = self.x_graph.clone()
        baseline_reshaped = self.baseline.reshape(1, -1)
        x_masked.x[~coalition_tensor] = baseline_reshaped
        return x_masked

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the GNN for each coalition by masking inactive node features."""
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
                coalition_value = model_output[0, self.y_index]
            else:
                coalition_value = model_output.squeeze()

            # Extrahiere den Scalar-Wert (für Tensoren oder NumPy-Arrays)
            if isinstance(coalition_value, torch.Tensor):
                coalition_values.append(coalition_value.item())  # .item() für Tensoren
            else:
                coalition_values.append(float(coalition_value))  # float() für NumPy-Skalare

        return np.array(coalition_values)
