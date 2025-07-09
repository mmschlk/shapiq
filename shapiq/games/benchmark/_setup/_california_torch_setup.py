"""Setup for the CaliforniaHousing dataset's neural network model.

Note:
    Note this should not be directly imported in the module as it depends on ``torch`` which is not
    installed by default.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.optim import Adam

if TYPE_CHECKING:
    import numpy as np

__all__ = ["CaliforniaHousingTorchModel"]


class SmallNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
            nn.Linear(5, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CaliforniaHousingTorchModel:
    def __init__(self, n_epochs: int = 100) -> None:
        # instantiate the model
        self.torch_model = SmallNeuralNetwork()
        try:
            self._load_torch_model_weights()
        except FileNotFoundError:
            # warn the user that the model could not be found and train a new model
            msg = (
                "The pre-trained neural network model for the CaliforniaHousing dataset could not "
                "be found. Please download the model from the shapiq repository. The model can be "
                "found at tests/models/california_nn_0.812511_0.076331.weights."
            )
            warnings.warn(msg, stacklevel=2)

        self.n_epochs = n_epochs

    def _torch_model_call(self, x: np.ndarray) -> np.ndarray:
        """A wrapper function to call the pre-trained neural network model on the numpy input data.

        Args:
            x: The input data to predict on.

        Returns:
            The model's prediction on the input data.

        """
        self.torch_model.eval()
        x = torch.tensor(x.astype(float), dtype=torch.float32)
        return self.torch_model(x).flatten().detach().numpy()

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Scores the model on the input data."""
        return r2_score(y, self._torch_model_call(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the output of the model on the input data."""
        return self._torch_model_call(x)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the model on the input data."""
        x = torch.tensor(x.astype(float), dtype=torch.float32)
        y = torch.tensor(y.astype(float), dtype=torch.float32)
        criterion = torch.nn.MSELoss()
        optimizer = Adam(self.torch_model.parameters(), lr=0.01)

        self.torch_model.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.torch_model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        self.torch_model.eval()

    def _load_torch_model_weights(self) -> None:
        """Loads a pre-trained neural network model for the CaliforniaHousing dataset."""
        # the file is located in the tests/data/models directory
        full_path = Path(__file__).resolve()
        parts = full_path.parts
        if "shapiq" in parts:
            idx = parts.index("shapiq")
            module_dir = Path(*parts[:idx])  # everything before shapiq
        else:
            msg = "shapiq not in path"
            raise ValueError(msg)
        path = Path("tests", "data", "models", "california_nn_0.812511_0.076331.weights")
        test_model_path = module_dir / "shapiq" / path
        self.torch_model.load_state_dict(torch.load(test_model_path))
        self.torch_model.eval()
