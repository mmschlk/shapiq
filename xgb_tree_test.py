# testfile to reconstruct issue 370: Instantiating TreeExplaliner object when XGB contains boosting tree without split
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from shapiq.explainer.tree import TreeExplainer

# Dummy data
X_train = np.array([[0], [1], [2], [3]])
y_train = np.array([0, 1, 0, 0])

# create model instance with no splits
model = XGBClassifier(n_estimators=1, max_depth=0, learning_rate=1, objective="binary:logistic")
# fit model
model.fit(X_train, y_train)


# evaluate using shapiq
explainer_shapiq_cl = TreeExplainer(
    model=model,
    max_order=1,
    index="SV",
    class_index=y_train,
)
sv_shapiq_cl = explainer_shapiq_cl.explain(x=X_train[0])
sv_shapiq_values_cl = sv_shapiq_cl.get_n_order_values(1)
