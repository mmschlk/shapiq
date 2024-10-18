"""This script trains a neural network on the California housing dataset."""

import copy
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shapiq.datasets import load_bike_sharing, load_california_housing

# Hyperparameters
BATCH_SIZE = 50
NUM_EPOCHS = 500
LEARNING_RATE = 0.0001


N_INPUT_CALIFORNIA = 8
N_INPUT_BIKE = 12


class CaliforniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(N_INPUT_CALIFORNIA, 100)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(50, 10)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(10, 1)

    def forward(self, x):
        return self.lin4(self.relu3(self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(x)))))))


class BikeSharingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(N_INPUT_BIKE, 100)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(50, 10)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(10, 1)

    def forward(self, x):
        return self.lin4(self.relu3(self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(x)))))))


class ScikitWrapper:

    def __init__(self, model_path):
        if "california" in model_path:
            self.model = CaliforniaModel()
        else:
            self.model = BikeSharingModel()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            try:
                # first make float numpy
                X = np.array(X).astype(np.float32)
                X = torch.tensor(X).float()
            except TypeError:
                raise TypeError("Input should be a torch tensor or a numpy array")
        pred = self.model(X).clone().detach()
        pred = pred.numpy()[:, 0]
        return pred


class BikeSharingScikitWrapper(ScikitWrapper):
    def __init__(self, model_path):
        super().__init__(model_path)


class CaliforniaScikitWrapper(ScikitWrapper):
    def __init__(self, model_path):
        super().__init__(model_path)


def train(
    california: bool,
    random_seed=1,
    save_checkpoint: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.25,
) -> tuple[Union[CaliforniaModel, BikeSharingModel], float, float]:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if california:
        save_name = f"california_model_{random_seed}.pth"
        X, y = load_california_housing(to_numpy=True, pre_processing=True)
    else:
        save_name = f"bike_model_{random_seed}.pth"
        X, y = load_bike_sharing(to_numpy=True)
        X = StandardScaler().fit_transform(X)
        # log normalize the target
        y = np.log(y + 1)

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

    model = CaliforniaModel() if california else BikeSharingModel()
    criterion = nn.MSELoss(reduction="sum")

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
    best_model = CaliforniaModel() if california else BikeSharingModel()
    best_model.load_state_dict(torch.load(save_name, weights_only=True))
    best_model.eval()
    outputs = best_model(X_test)
    err = np.sqrt(mean_squared_error(outputs.detach().numpy(), y_test.detach().numpy()))
    r2 = r2_score(outputs.detach().numpy(), y_test.detach().numpy())
    print(f"\nTest Error: {err}, Test R2 Score: {r2}\n")
    return best_model, err, r2


def evaluate(
    california: bool, random_seed=1, test_size: float = 0.2
) -> tuple[Union[CaliforniaScikitWrapper, BikeSharingScikitWrapper], float, float]:
    if california:
        model_path = f"california_model_{random_seed}.pth"
        X, y = load_california_housing(to_numpy=True, pre_processing=True)
    else:
        model_path = f"bike_model_{random_seed}.pth"
        X, y = load_bike_sharing(to_numpy=True)
        X = StandardScaler().fit_transform(X)
        y = np.log(y + 1)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).view(-1, 1).float()

    if california:
        model = CaliforniaScikitWrapper(model_path)
    else:
        model = BikeSharingScikitWrapper(model_path)

    outputs = model.predict(X_test)
    err = np.sqrt(mean_squared_error(outputs, y_test.detach().numpy()))
    r2 = r2_score(outputs, y_test.detach().numpy())
    print(f"\nTest Error: {err}, Test R2 Score: {r2}\n")
    return model, err, r2


if __name__ == "__main__":

    train_california = True  # False for Titanic

    random_seed = 42
    k_folds = 5
    do_k_fold = True
    do_train = False
    do_test = False

    if do_k_fold:
        # do k-fold monte-carlo cross validation
        errors, r2s = [], []
        for i in range(k_folds):
            model, err, r2 = train(california=train_california, random_seed=i, save_checkpoint=True)
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

        # save the results to a file
        with open("california_k_fold.txt", "w") as f:
            f.write(f"Errors: {errors}\n")
            f.write(f"R2s: {r2s}\n")
            f.write(f"Mean Error: {error_mean}, Std Error: {error_std}\n")
            f.write(f"Mean R2: {r2_mean}, Std R2: {r2_std}\n")

    if do_train:
        train(
            california=train_california,
            random_seed=random_seed,
            save_checkpoint=True,
            test_size=0.3,
            val_size=0.2,
        )

    if do_test:
        evaluate(
            california=train_california,
            random_seed=random_seed,
            test_size=0.3,
        )
