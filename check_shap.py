"""This script checks if the SV computed for an IsoForest model are the same for shap and shapiq."""

import copy
from dataclasses import dataclass

import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import shapiq


def generate_random_perturb_features(settings):
    """
    NOTE: FOR TESTING PURPOSES ONLY
    Generates a set of clusters and perturbs a RANDOM subset of features to create outliers.
    """
    # Define parameters
    n_samples = settings.n_samples
    n_outliers = settings.n_outliers
    n_clusters = settings.n_clusters
    n_features = settings.n_features
    n_perturb_features = settings.n_perturb_features
    always_perturb_same_features = settings.always_perturb_same_features
    random_state = settings.random_state

    rng = np.random.RandomState(random_state)

    # Create random mean vectors for the clusters, ensuring they are different
    means = []
    for _ in range(n_clusters):
        mean = rng.rand(n_features) * 4 - 2
        while any(np.linalg.norm(mean - m) < 2 for m in means):
            mean = rng.rand(n_features) * 4 - 2
        means.append(mean)

    # Generate clusters with different covariance matrices
    clusters = []
    for mean in means:
        # Create a random covariance matrix for the desired number of features
        covariance = rng.rand(n_features, n_features)
        covariance = covariance @ covariance.T  # Make it symmetric and positive semi-definite
        cluster = 0.4 * rng.randn(n_samples, n_features) @ covariance + mean
        clusters.append(cluster)

    # Combine clusters
    data = np.vstack(clusters)

    # Function to perturb a subset of features to create outliers
    def create_outliers(
        data, n_outliers, n_features, n_perturb_features, always_perturb_same_features
    ):
        original_indices = rng.choice(len(data), n_outliers, replace=False)
        outliers = data[original_indices].copy()

        if always_perturb_same_features:
            features_to_perturb = rng.choice(n_features, n_perturb_features, replace=False)

        for i in range(n_outliers):
            if not always_perturb_same_features:
                features_to_perturb = rng.choice(n_features, n_perturb_features, replace=False)
            # Perturb the selected features
            outliers[i, features_to_perturb] += rng.normal(0, 5, n_perturb_features)
        return outliers, original_indices

    # Create outliers and get the indices of the original points
    outliers, original_indices = create_outliers(
        data, n_outliers, n_features, n_perturb_features, always_perturb_same_features
    )

    # Store original samples and perturbed samples separately
    original_samples = data[original_indices].copy()
    perturbed_samples = outliers.copy()

    # Remove the original points that were perturbed from the dataset
    data = np.delete(data, original_indices, axis=0)

    # Combine inliers and outliers
    final_data = np.vstack([data, outliers])

    # Create labels
    labels = np.ones(len(final_data))
    labels[-n_outliers:] = -1

    # print("Original Samples:\n", original_samples)
    # print("Perturbed Samples (Outliers):\n", perturbed_samples)
    # print("Final Data:\n", final_data)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        final_data, labels, test_size=0.33, random_state=42
    )

    # Create a mapping between original points and their perturbed versions
    original_to_perturbed = {i: i for i in range(n_outliers)}
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        original_samples,
        perturbed_samples,
        original_to_perturbed,
    )


@dataclass
class SyntheticOutlierInlierSettings:
    n_samples: int = 1000
    n_outliers: int = 100
    n_clusters: int = 2
    n_features: int = 12
    n_perturb_features: int = 2
    always_perturb_same_features: bool = True
    noise_level: float = 5
    random_state: int = None


if __name__ == "__main__":

    # create data
    settings = SyntheticOutlierInlierSettings(
        n_samples=120,
        n_outliers=40,
        n_clusters=1,
        n_features=12,
        n_perturb_features=2,
        always_perturb_same_features=True,
        random_state=0,
    )
    d = generate_random_perturb_features(settings)
    X_train, X_test, y_train, y_test, original_samples, perturbed_samples, original_to_perturbed = d

    # train model
    clf = IsolationForest(max_samples=100, random_state=0)
    clf.fit(X_train)

    x_explain = copy.deepcopy(X_test[0])
    print(x_explain)

    # explain with shap
    explainer = shap.TreeExplainer(clf)
    sv_shap = explainer.shap_values(x_explain)
    print(sv_shap)

    # explain with shapiq
    explainer = shapiq.TreeExplainer(clf, index="SV", max_order=1)
    sv_shapiq = explainer.explain(x_explain)
    print(sv_shapiq)

    for i, sv in enumerate(sv_shap):
        sv_iq = sv_shapiq[(i,)]
        print(sv, sv_iq)
        assert np.allclose(sv, sv_iq)
