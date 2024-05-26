"""Note this should not be directly imported in the module as it depends on torch which is not
installed by default."""

import os
import warnings

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.optim import Adam


class SmallNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CaliforniaHousingTorchModel:

    def __init__(self, n_epochs: int = 100):

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
            warnings.warn(msg)

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
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.torch_model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        self.torch_model.eval()

    def _load_torch_model_weights(self) -> None:
        """Loads a pre-trained neural network model for the CaliforniaHousing dataset."""
        # the file is located in the tests/data/models directory
        module_dir = os.path.abspath(__file__).split("shapiq")[0]
        path = os.path.join("tests", "data", "models", "california_nn_0.812511_0.076331.weights")
        test_model_path = os.path.join(module_dir, "shapiq", path)
        self.torch_model.load_state_dict(torch.load(test_model_path))
        self.torch_model.eval()
