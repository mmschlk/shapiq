"""This is a shorthand example of how to use the shapiq package to approximate Shapley values for
a small scale tabular dataset."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import shapiq
from shapiq.games.imputer import MarginalImputer

if __name__ == "__main__":

    xb_model: bool = True

    data, target = shapiq.datasets.load_california_housing()
    features = data.columns  # list of features
    n_features = len(features)  # number of features (also players)
    data, target = data.values, target.values  # get arrays

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )

    # fit a model
    if xb_model:
        model = XGBRegressor(n_estimators=10, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(data, target)
    print(f"Train R2: {model.score(X_train, y_train):.4f}")
    print(f"Val R2: {model.score(X_test, y_test):.4f}")

    # select a data point to explain
    x = X_test[0]

    # create a game / set valued function of the model
    imputer = MarginalImputer(
        model=model.predict, data=X_train, x=x, sample_replacements=True, sample_size=5
    )

    # compute the shapley values with KernelSHAP
    approximator = shapiq.approximator.KernelSHAP(n=n_features, random_state=42)
    sv_values = approximator.approximate(budget=2**n_features, game=imputer)

    # print the Shapley values
    print(sv_values)

    # get interactions
    interaction_approximator = shapiq.approximator.ShapIQ(
        n=n_features, max_order=2, index="k-SII", random_state=42
    )
    interaction_values = interaction_approximator.approximate(budget=2**n_features, game=imputer)

    # print the interaction values
    print(interaction_values)
