"""This script demonstrates how to use the ProductKernelExplainer with both regression and classification models."""

from __future__ import annotations

from sklearn.datasets import make_classification, make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from shapiq.explainer.product_kernel.explainer import ProductKernelExplainer

# Generate a synthetic regression dataset with 10 features
X, y = make_regression(n_samples=200, n_features=10, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Test instance
x = X_test[0]  # Instance to explain

# Train an SVR model with RBF kernel
svr_model = SVR(kernel="rbf", C=1.0, gamma="scale")
svr_model.fit(X_train, y_train)

# Initialize the explainer with this model
svr_explainer = ProductKernelExplainer(svr_model, max_order=2, min_order=1, index="k-SII")

# Compute Shapley values
svr_shapley_values = svr_explainer.explain(x)
print("SVR Shapley Values:", svr_shapley_values)  # noqa: T201
print(f"sum of SVR Shapley Values is: {sum(svr_shapley_values)}")  # noqa: T201
# using sklearn functions:
print(f"the intercept in the model is: {svr_model.intercept_}")  # noqa: T201
print(f"Prediction: {svr_model.predict([x])[0]}")  # noqa: T201


# train a GP
kernel = RBF(1.0, (1e-3, 1e3))
gp_reg_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gp_reg_model.fit(X_train, y_train)

gp_reg_explainer = ProductKernelExplainer(gp_reg_model)
gp_reg_shapley_values = gp_reg_explainer.explain(x)
print("GP Shapley Values:", gp_reg_shapley_values)  # noqa: T201
print(f"sum of GP Shapley Values is {sum(gp_reg_shapley_values)}")  # noqa: T201


# ------------------------- Classification Example -------------------------
# Generate a synthetic classification dataset (binary classification)
X_clf, y_clf = make_classification(
    n_samples=200, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42
)  # <sup data-citation="6" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/">6</a></sup>

# Split the data into training and testing sets
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Standardize the features
scaler_clf = StandardScaler()
X_train_clf = scaler_clf.fit_transform(X_train_clf)
X_test_clf = scaler_clf.transform(X_test_clf)

# Test instance for classification
x_clf = X_test_clf[0]  # instance to explain


# Train an SVC model with RBF kernel (set probability=True to enable probability estimates)
svc_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42)
svc_model.fit(
    X_train_clf, y_train_clf
)  # <sup data-citation="4" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/">4</a></sup>

# Initialize the explainer with the chosen classifier (e.g., the GP classifier)
svc_explainer = ProductKernelExplainer(svc_model)  # use same explainer interface as for regression

# Compute Shapley values for classification
svc_shapley_values = svc_explainer.explain(x_clf)
print("SVC Shapley Values (Classification):", svc_shapley_values)  # noqa: T201
# You can also observe the predicted probability and the predicted class:
print(f"sum of SVC Shapley Value {sum(svc_shapley_values)}")  # noqa: T201
print("predicted decision function: ", svc_model.decision_function([x_clf])[0])  # noqa: T201
print("intercept is: ", svc_model.intercept_)  # noqa: T201

"""
# Alternatively, train a Gaussian Process Classifier with an RBF kernel
kernel = RBF(1.0, (1e-3, 1e3))
gp_clf_model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp_clf_model.fit(X_train_clf,
                 y_train_clf)  # <sup data-citation="6" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/">6</a></sup>

# Initialize the explainer with the chosen classifier (e.g., the GP classifier)
gp_clf_explainer = ProductKernelExplainer(gp_clf_model)  # use same explainer interface as for regression


# Compute Shapley values for classification
gp_clf_shapley_values = gp_clf_explainer.explain(x_clf)
print("GP Shapley Values (Classification):", gp_clf_shapley_values)
print(f"sum of GP Shapley Value {sum(gp_clf_shapley_values)}")
# You can also observe the predicted probability and the predicted class:
print("Predicted probabilities:", gp_clf_model.predict_proba([x_clf])[0])
print("Predicted class:", gp_clf_model.predict([x_clf])[0])
"""
