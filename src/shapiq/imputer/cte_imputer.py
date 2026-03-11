"""Implementation of the marginal imputer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from goodpoints import compress

from .base import Imputer

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues, Model


class CTEImputer(Imputer):
    """The compress then explain (CTE) imputer for the shapiq package.

    The CTE imputer replaces missing features of the explanation point ``x`` by values
    sampled from the background data. Background data is first subsampled using a distribution
    compression algorithm, and then rows are sampled jointly from the compressed background data.
    This has shown to provide accurate and stable estimates of explanations while being computationally 
    efficient. For details, see the paper introducing CTE by Baniecki et al. (2025) [Ban25]_.

    This corresponds to *interventional* imputation (often called *marginal fANOVA* in the
    literature), as opposed to *observational* imputers that condition on observed features.

    Examples:
        >>> model = lambda x: np.sum(x, axis=1)  # some dummy model
        >>> data = np.random.rand(1000, 4)  # some background data
        >>> x_to_impute = np.array([[1, 1, 1, 1]])  # some data point to impute
        >>> imputer = CTEImputer(model=model, data=data, x=x_to_impute, random_state=42)
        >>> # get the model prediction with missing values
        >>> imputer(np.array([[True, False, True, False]]))
        np.array([2.01])  # some model prediction (might be different)
        >>> # exchange the background data
        >>> new_data = np.random.rand(1000, 4)
        >>> imputer.init_background(data=new_data)

    See Also:
        - :class:`shapiq.imputer.MarginalImputer` for the marginal imputer.
        - :class:`shapiq.imputer.BaselineImputer` for the baseline imputer.
        - :class:`shapiq.imputer.base.Imputer` for the base imputer class.

    References:
        .. [Ban25] Baniecki, H., Casalicchio, G., Bischl, B., Biecek, P., (2025). Efficient and Accurate Explanation Estimation with Distribution Compression. In International Conference on Learning Representations. url: https://openreview.net/forum?id=LiUfN9h0Lx

    """


    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        *,
        x: np.ndarray | None = None,
        normalize: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initializes the marginal imputer.

        Args:
            model: The model to explain as a callable function expecting a data points as input and
                returning the model's predictions.

            data: The background data to use for the explainer as a two-dimensional array
                with shape ``(n_samples, n_features)``.

            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape ``(1, n_features)`` or as a vector with shape ``(n_features,)``. If ``None``,
                the imputer must be fitted before it can be used.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features.

            random_state: The random state to use for sampling. If ``None``, the random state is not
                fixed.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            random_state=random_state,
        )

        # setup attributes
        self._replacement_data: np.ndarray = np.zeros((1, self.n_features))
        self.init_background(self.data)

        if normalize:  # update normalization value
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Imputes the missing values of a data point and calls the model.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which
                are missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
               ``(n_subsets, n_outputs)``.

        """
        n_coalitions = coalitions.shape[0]
        sample_size = self._replacement_data.shape[0]
        outputs = np.zeros((sample_size, n_coalitions))
        imputed_data = np.tile(self.x, (n_coalitions, 1))
        for i in range(sample_size):
            replacements = np.tile(self._replacement_data[i], (n_coalitions, 1))
            imputed_data[~coalitions] = replacements[~coalitions]
            predictions = self.predict(imputed_data)
            outputs[i] = predictions
        outputs = np.mean(outputs, axis=0)  # average over the samples
        # insert the better approximate empty prediction for the empty coalitions
        outputs[~np.any(coalitions, axis=1)] = self.empty_prediction
        return outputs

    def init_background(self, data: np.ndarray) -> CTEImputer:
        """Initializes the imputer to a background data set.

        The background data is used to sample replacement values for the missing features. To change
        the background data, use this method.

        Args:
            data: The background data to use for the imputer. The shape of the array must
                be ``(n_samples, n_features)``.

        Returns:
            The initialized imputer.

        Examples:
            >>> model = lambda x: np.sum(x, axis=1)
            >>> data = np.random.rand(10, 3)
            >>> imputer = MarginalImputer(model=model, data=data, x=data[0])
            >>> new_data = np.random.rand(10, 3)
            >>> imputer.init_background(data=new_data)

        Raises:
            UserWarning: If the sample size is larger than the number of data points in the
                background data. In this case, the sample size is reduced to the number of data
                points in the background data.

        """
        d = data.shape[1]
        sigma = np.sqrt(2 * d)
        id_compressed = compress.compresspp_kt(data, kernel_type=b"gaussian", k_params=np.array([sigma**2]), g=4, seed=self.random_state)
        self._replacement_data = data[id_compressed]
        self.calc_empty_prediction()  # reset the empty prediction to the new background data
        return self 

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction of the model provided only missing features.

        """
        empty_predictions = self.predict(self._replacement_data)
        empty_prediction = float(np.mean(empty_predictions))
        self.empty_prediction = empty_prediction
        if self.normalize:  # reset the normalization value
            self.normalization_value = empty_prediction
        return empty_prediction
