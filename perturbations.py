import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import shapiq
from shapiq.games.imputer import BaselineImputer, ConditionalImputer, MarginalImputer


def train_ch():
    X, y = shapiq.load_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.25, random_state=42
    )
    n_features = X_train.shape[1]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=n_features,
        max_features=2 / 3,
        max_samples=2 / 3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"Train R2: {model.score(X_train, y_train):.4f}")
    print(f"Test  R2: {model.score(X_test, y_test):.4f}")

    return model, X_train, y_train, X_test, y_test


if __name__ == "__main__":

    # imputer = SentimentAnalysisLocalXAI("This is a great movie!")
    # imputer = ImageClassifierLocalXAI(
    #    x_explain_path="docs/source/notebooks/vision_notebooks/original_image.jpg"
    # )

    model, X_train, Y_train, X_test, y_test = train_ch()

    # IMPUTE_MODE = "baseline"
    IMPUTE_MODE = "marginal"
    # IMPUTE_MODE = "conditional"
    if IMPUTE_MODE == "baseline":
        b = np.mean(X_train, 0)
        imputer = BaselineImputer(model.predict, b, X_train[0, :])
    if IMPUTE_MODE == "marginal":
        n_background = 128
        background_data = X_train[:n_background, :]
        np.mean(model.predict(background_data))
        imputer = MarginalImputer(
            model.predict,
            background_data,
            X_train[0, :],
            sample_size=n_background,
            normalize=False,
            sample_replacements=False,
        )
    if IMPUTE_MODE == "conditional":
        # not really ready
        imputer = ConditionalImputer(model.predict, X_train[:128, :], X_train[0, :])

    n_players = imputer.n_players
    all_mask = np.zeros(n_players, dtype=bool)
    no_mask = np.ones(n_players, dtype=bool)
    print("Prediction (full set): ", imputer(no_mask))
    print("Masked Prediction (emptyset): ", imputer(all_mask))

    random_masks = np.random.randint(0, 2, (1000, n_players), dtype=bool)
    print("Prediction on Random Masks: ", imputer(random_masks))
