"""Interventional game for tree-based models."""

from __future__ import annotations

import numpy as np

from shapiq.game import Game
from shapiq.utils.modules import safe_isinstance


class InterventionalGame(Game):
    """A cooperative game for interventional tree-based model explanations."""

    def __init__(
        self,
        model: object,
        reference_data: np.ndarray,
        target_instance: np.ndarray,
        class_index: int | None = None,
    ) -> None:
        """Initialize the InterventionalGame.

        Args:
            model: The tree-based model to explain.
            reference_data: Background dataset used as reference.
            target_instance: The instance to explain.
            class_index: Class index for classification models. Defaults to ``None``.
        """
        if target_instance.ndim == 1:
            target_instance = target_instance.reshape(1, -1)
        super().__init__(
            n_players=target_instance.shape[1], normalize=False, normalization_value=0
        )  # number of features

        # Set class index if classification model
        if hasattr(model, "predict_proba") and class_index is None:
            class_index = 1  # default to positive class for binary classification
        self.model = model
        self.data = reference_data
        self.target_instance = target_instance
        self.class_index = class_index

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute the value function for the given coalitions.

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_players).

        Returns:
            Array of values for each coalition.
        """
        n_coalitions = coalitions.shape[0]
        values = np.zeros(n_coalitions)
        for i in range(n_coalitions):
            coalition = coalitions[i]
            vls = None
            instanceses = np.where(coalition, self.target_instance, self.data)
            if self.class_index is not None:
                if safe_isinstance(self.model, "xgboost.sklearn.XGBClassifier"):
                    import xgboost as xgb

                    # For XGBClassifier, we need to use DMatrix for prediction with output_margin
                    dmatrix_instance = xgb.DMatrix(instanceses)
                    booster = self.model.get_booster()  # ty: ignore[unresolved-attribute]
                    logits = booster.predict(dmatrix_instance, output_margin=True)
                    # Append the logit for the specified class index
                    if logits.ndim == 1:
                        # Binary classification case
                        vls = logits if self.class_index == 1 else -logits
                    else:
                        vls = logits[:, self.class_index]
                elif safe_isinstance(self.model, "lightgbm.LGBMClassifier"):
                    proba = self.model.predict_proba(  # ty: ignore[unresolved-attribute]
                        instanceses
                    )
                    vls = proba[:, self.class_index]
                    logit = np.log(vls / (1 - vls))
                    vls = logit
                else:
                    proba = self.model.predict_proba(  # ty: ignore[unresolved-attribute]
                        instanceses
                    )
                    vls = proba[:, self.class_index]
            else:
                vls = self.model.predict(instanceses)  # ty: ignore[unresolved-attribute]

            values[i] = np.mean(vls)
        return values
