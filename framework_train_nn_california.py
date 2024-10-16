"""This script trains a neural network on the California housing dataset."""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shapiq.datasets import load_california_housing

torch.manual_seed(1234)
np.random.seed(1234)


batch_size = 50
num_epochs = 100
learning_rate = 0.0001
size_hidden1 = 100
size_hidden2 = 50
size_hidden3 = 10
size_hidden4 = 1


class CaliforniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, size_hidden1)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(size_hidden1, size_hidden2)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(size_hidden2, size_hidden3)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(size_hidden3, size_hidden4)

    def forward(self, input):
        return self.lin4(self.relu3(self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(input)))))))


def train(model_inp):
    optimizer = torch.optim.RMSprop(model_inp.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in train_loader:
            # forward pass
            outputs = model_inp(inputs)
            # defining loss
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            # accumulating running loss
            running_loss += loss.item()
            # updated weights based on computed gradients
            optimizer.step()
        if epoch % 2 == 0:
            print(
                "Epoch [%d]/[%d] running accumulative loss across all batches: %.3f"
                % (epoch + 1, num_epochs, running_loss)
            )
        running_loss = 0.0


def train_load_save_model(model_obj, model_path, retrain=False):
    if os.path.exists(model_path):
        # load model
        print(f"Loading pre-trained model from: {model_path}")
        model_obj.load_state_dict(torch.load(model_path))
        if retrain:
            model_obj.train()
            train(model_obj)
            print(f"Finished re-training the model. Saving the model to the path: {model_path}")
            torch.save(model_obj.state_dict(), model_path)
    else:
        # train model
        train(model_obj)
        print(f"Finished training the model. Saving the model to the path: {model_path}")
        torch.save(model_obj.state_dict(), model_path)


class CaliforniaScikitWrapper:

    def __init__(self, model_path):
        model = CaliforniaModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model = model

    def predict(self, X):
        return self.model(torch.tensor(X).float()).detach().numpy()[:, 0]


if __name__ == "__main__":
    X, y = load_california_housing(to_numpy=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).view(-1, 1).float()

    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).view(-1, 1).float()

    datasets = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=50, shuffle=True)

    # parameter
    model = CaliforniaModel()
    model.train()

    criterion = nn.MSELoss(reduction="sum")

    SAVED_MODEL_PATH = "california_model.pt"
    train_load_save_model(model, SAVED_MODEL_PATH, retrain=True)

    model.eval()
    outputs = model(X_test)
    err = np.sqrt(mean_squared_error(outputs.detach().numpy(), y_test.detach().numpy()))
    r2 = r2_score(outputs.detach().numpy(), y_test.detach().numpy())

    print("model err: ", err, "model r2: ", r2)
