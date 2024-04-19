import shapiq

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = shapiq.load_bike_sharing()
X, y = data.drop("Count", axis=1).values, data.Count.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
n_features = X.shape[1]

model = RandomForestRegressor(n_estimators=100, max_depth=3, max_features="sqrt", random_state=42)
model.fit(X_train, y_train)
print("Train R2: {:.4f}".format(model.score(X_train, y_train)))
print("Val R2: {:.4f}".format(model.score(X_test, y_test)))

explainer_tree = shapiq.TreeExplainer(model=model, interaction_type="k-SII", max_order=2)

x = X_test[0]

interaction_values_tree = explainer_tree.explain(x)

interaction_values_tree
