"""This module contains the base TreeSHAP-IQ xai game."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from shapiq.explainer.tree import TreeExplainer, TreeModel
from shapiq.games.base import Game

from shap import TreeExplainer as TreeExplainerSHAP

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from shapiq.utils.custom_types import Model


class TreeSHAPInterventionalXAI(Game):
    """A interventional TreeSHAP explanation game for tree models.

    The game is based on the shap TreeSHAP algorithm using interventional perturbations and is used to explain the predictions of tree
    models. TreeSHAP is used to compute Shapley values for tree models.
    """

    def __init__(
        self,
        x: np.ndarray,
        tree_model: Model,
        *,
        class_label: int | None = None,
        normalize: bool = True,
        verbose: bool = True,
        feature_perturbation: str = "tree_path_dependent",
        background_data: np.ndarray | None = None,
    ) -> None:
        """Initializes the interventional TreeSHAP explanation game.

        Args:
            x: The feature vector to be explained. If ``None``, then the first data point is used.
                If an integer, then the data point at the given index is used. If a numpy array,
                then the data point is used as is. Defaults to ``None``.

            tree_model: The tree model to explain as a callable function. The model must be a
                decision tree or random forest model.

            class_label: The class label to use for the model. If ``None``, then the model is
                assumed to be a regression model and / or the default behaviour of
                :class:`~shapiq.explainer.tree.TreeExplainer` is used.

            normalize: A boolean flag to normalize/center the game values. The default value is
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.
        """
        n_players = x.shape[-1]

        self.model = copy.deepcopy(tree_model)
        self.class_label = class_label

        # set up explainer for model transformation
        self._tree_explainer = TreeExplainerSHAP(
            model=tree_model,
            feature_perturbation=feature_perturbation,
            data=background_data,
        )

        # get attributes for manual tree traversal and evaluation
        self.x_explain = x
        # transform x_explain into a 1-dimensional array if it is a 2-dimensional array
        if x.ndim == 2 and x.shape[0] == 1:
            self.x_explain = x.flatten()
        elif x.ndim > 2:
            msg = "x_explain must be a 1-dimensional or 2-dimensional array."
            raise ValueError(msg)

        # compute ground truth values
        empty_value_allclasses = self._tree_explainer.expected_value
        shapley_values_allclasses = self._tree_explainer.shap_values(self.x_explain)

        if class_label is not None:
            self.empty_value = empty_value_allclasses[class_label]
            self.shapley_values = shapley_values_allclasses[:,class_label]
        else:
            self.empty_value = empty_value_allclasses
            self.shapley_values = shapley_values_allclasses

        # set up game
        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the output of the tree model given coalitions of features.

        Args:
            coalitions: A binary matrix of feature subsets present (`True`) or absent (`False`).
                The matrix is expected to be in shape `(n_coalitions, n_features)`.

        Returns:
            The worth of the coalitions as a vector.

        """
        worth = np.zeros(len(coalitions), dtype=float)
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:  # empty subset
                worth[i] = self.empty_value
                continue
            worth[i] = self.compute_tree_output_from_coalition(coalition)
        return worth

    def compute_tree_output_from_coalition(self, coalition: np.ndarray) -> float:
        """Computes the output of the tree models contained in the game.

        Args:
            coalition: A binary vector of feature indices present (`True`) and absent (`False`) from
                the coalition.

        Returns:
            The prediction given partial feature information as an average of individual tree
                predictions.

        """
        # impute data
        imputation_data = self._tree_explainer.data
        x_explain = self.x_explain
        if x_explain.ndim == 1:
            # if x_explain is a 1-dimensional array, reshape it to 2D where the first dimensions is 1
            x_explain = x_explain.reshape(1, -1)
        n_imputations = imputation_data.shape[0]
        n_explanations = x_explain.shape[0]
        # Repeat data (block-wise)
        data_repeated = np.repeat(imputation_data, n_explanations, axis=0)
        # Tile x_explain n times (cycle-wise)
        x_explain_tiled = np.tile(x_explain, (n_imputations, 1))

        imputed_data = data_repeated.copy()
        imputed_data[:, coalition] = x_explain_tiled[:, coalition]

        # call mode
        predictions = self._tree_explainer.model.predict(imputed_data)
        if self.class_label is not None:
            # if class_label is set, we only take the prediction for the class_label
            predictions = predictions[:, self.class_label]
        predictions = predictions.reshape(n_imputations, n_explanations)
        output = np.mean(predictions, axis=0)

        return output

    def exact_values(self, index: str, order: int) -> InteractionValues:
        """Computes the exact interaction values for the game.

        Args:
            index: The type of interaction index to be computed.
            order: The order of interactions to be computed.

        Returns:
            The exact interaction values for the game.

        """
        if index == "SV" and order == 1:
            return self.shapley_values
        else:
            msg = f"Exact values for index {index} and order {order} are not implemented."
            raise NotImplementedError(msg)