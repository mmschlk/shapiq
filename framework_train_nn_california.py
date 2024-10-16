"""This script trains a neural network on the California housing dataset."""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shapiq.datasets import load_california_housing

torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
BATCH_SIZE = 50
NUM_EPOCHS = 1000
LEARNING_RATE = 0.0001

SIZE_HIDDEN1 = 100
SIZE_HIDDEN2 = 50
SIZE_HIDDEN3 = 10


class CaliforniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, SIZE_HIDDEN1)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(SIZE_HIDDEN1, SIZE_HIDDEN2)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(SIZE_HIDDEN2, SIZE_HIDDEN3)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(SIZE_HIDDEN3, 1)

    def forward(self, input):
        return self.lin4(self.relu3(self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(input)))))))


def train(
    random_seed=1, save_checkpoint: bool = True, test_size: float = 0.2, val_size: float = 0.25
) -> tuple[CaliforniaModel, float, float]:
    save_name = f"california_model_{random_seed}.pth"

    X, y = load_california_housing(to_numpy=True)
    # makes a train-test-validation split of 60-20-20 with test_size=0.2 and val_size=0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True,
    )

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).view(-1, 1).float()

    X_val = torch.tensor(X_val).float()
    y_val = torch.tensor(y_val).view(-1, 1).float()

    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).view(-1, 1).float()

    datasets = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)

    model = CaliforniaModel()
    model.train()

    criterion = nn.MSELoss(reduction="sum")

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = np.inf
    best_model = model
    print(f"Training model with random seed: {random_seed}")
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in train_loader:
            # forward pass
            outputs = model(inputs)
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

        # validate the model
        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        model.train()

        if epoch % 2 == 0:
            print(f"Epoch: {epoch}, Training Loss: {running_loss}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss or epoch == 0:
            print(f"Best model at epoch: {epoch}")
            best_val_loss = val_loss
            if save_checkpoint:
                torch.save(model.state_dict(), save_name)
            best_model = copy.deepcopy(model)

    # always save the best model
    torch.save(best_model.state_dict(), save_name)
    best_model = CaliforniaModel()
    best_model.load_state_dict(torch.load(save_name, weights_only=True))

    # evaluation
    best_model.eval()
    outputs = best_model(X_test)
    err = np.sqrt(mean_squared_error(outputs.detach().numpy(), y_test.detach().numpy()))
    r2 = r2_score(outputs.detach().numpy(), y_test.detach().numpy())
    print(f"\nTest Error: {err}, Test R2 Score: {r2}\n")
    return best_model, err, r2


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

    k_folds = 5
    do_k_fold = True
    do_train = True

    if do_k_fold:
        # do k-fold monte-carlo cross validation
        errors, r2s = [], []
        for i in range(k_folds):
            model, err, r2 = train(random_seed=i, save_checkpoint=True)
            errors.append(err)
            r2s.append(r2)
        print(f"Errors: {errors}")
        print(f"R2s: {r2s}")

        error_mean = np.mean(errors)
        error_std = np.std(errors)
        print(f"Mean Error: {error_mean}, Std Error: {error_std}")

        r2_mean = np.mean(r2s)
        r2_std = np.std(r2s)
        print(f"Mean R2: {r2_mean}, Std R2: {r2_std}")

    if do_train:
        train(random_seed=42, save_checkpoint=True, test_size=0.3, val_size=0.2)
