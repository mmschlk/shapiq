from shapiq.tree.validation import validate_tree_model
from shapiq.game import Game
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from shapiq.utils.modules import safe_isinstance


class InterventionalGame(Game):
    def __init__(self, model, reference_data, target_instance, class_index=None):
        if target_instance.ndim == 1:
            target_instance = target_instance.reshape(1, -1)
        super().__init__(
            n_players=target_instance.shape[1], normalize=False, normalization_value=0
        )  # number of features

        # Set class index if classification model
        if hasattr(model, "predict_proba") and class_index is None:
            print(
                "No class index provided for classification model. Using default class index 1."
            )
            class_index = 1  # default to positive class for binary classification
        self.model = model
        self.data = reference_data
        self.target_instance = target_instance
        self.class_index = class_index

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        n_coalitions = coalitions.shape[0]
        values = np.zeros(n_coalitions)
        for i in range(n_coalitions):
            coalition = coalitions[i]
            vls = None
            instanceses = np.where(coalition, self.target_instance, self.data)
            if self.class_index is not None:
                if safe_isinstance(self.model, "xgboost.sklearn.XGBClassifier"):
                    # For XGBClassifier, we need to use DMatrix for prediction with output_margin
                    dmatrix_instance = xgb.DMatrix(instanceses)
                    logits = self.model.get_booster().predict(
                        dmatrix_instance, output_margin=True
                    )
                    # print("Logits:", logits)
                    # Append the logit for the specified class index
                    if logits.ndim == 1:
                        # Binary classification case
                        if self.class_index == 1:
                            vls = logits
                        else:
                            vls = -logits
                    else:
                        vls = logits[:, self.class_index]
                elif safe_isinstance(self.model, "lightgbm.LGBMClassifier"):
                    vls = self.model.predict_proba(instanceses)[:, self.class_index]
                    logit = np.log(vls / (1 - vls))
                    vls = logit
                else:
                    vls = self.model.predict_proba(instanceses)[:, self.class_index]                    
            else:
                vls = self.model.predict(instanceses)

            values[i] = np.mean(vls)
        return values
