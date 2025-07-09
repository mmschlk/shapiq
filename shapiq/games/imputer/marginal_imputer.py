"""Implementation of the marginal imputer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .base import Imputer

if TYPE_CHECKING:
    from shapiq.utils import Model

_too_large_sample_size_warning = (
    "The sample size is larger than the number of data points in the background set. "
    "Reducing the sample size to the number of background samples."
)


class MarginalImputer(Imputer):
    """The marginal imputer for the shapiq package.

    The marginal imputer is used to impute the missing values of a data point by using the
    marginal distribution of the background data.

    Attributes:
        replacement_data: The data to use for imputation. To change the data, use the
            ``init_background`` method.
        joint_marginal_distribution: A flag weather replacement values are sampled from the joint
            marginal distribution (``True``) or independently for each feature (``False``).
        empty_prediction: The model's prediction on an empty data point (all features missing).

    Examples:
        >>> model = lambda x: np.sum(x, axis=1)  # some dummy model
        >>> data = np.random.rand(1000, 4)  # some background data
        >>> x_to_impute = np.array([[1, 1, 1, 1]])  # some data point to impute
        >>> imputer = MarginalImputer(model=model, data=data, x=x_to_impute, sample_size=100)
        >>> # get the model prediction with missing values
        >>> imputer(np.array([[True, False, True, False]]))
        np.array([2.01])  # some model prediction (might be different)
        >>> # exchange the background data
        >>> new_data = np.random.rand(1000, 4)
        >>> imputer.init_background(data=new_data)

    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        *,
        x: np.ndarray | None = None,
        sample_size: int = 100,
        categorical_features: list[int] | None = None,
        joint_marginal_distribution: bool = True,
        normalize: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initializes the marginal imputer.

        Args:
            model: The model to explain as a callable function expecting a data points as input and
                returning the model's predictions.

            data: The background data to use for the explainer as a two-dimensional array
                with shape ``(n_samples, n_features)``.

            x: The explanation point to use the imputer to.

            sample_size: The number of samples to draw from the background data. Only used if
                ``sample_replacements`` is ``True``. Increasing this value will linearly increase
                the runtime of the explainer. Defaults to ``100``.

            categorical_features: A list of indices of the categorical features. If ``None``, all
                features are treated as continuous. Defaults to ``None``.

            joint_marginal_distribution: A flag to sample the replacement values from the joint
                marginal distribution. If ``False``, the replacement values are sampled
                independently for each feature. If ``True``, the replacement values are sampled from
                the joint marginal distribution. Defaults to ``False``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            random_state: The random state to use for sampling. Defaults to ``None``.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        # setup attributes
        self.joint_marginal_distribution: bool = joint_marginal_distribution
        self.replacement_data: np.ndarray = np.zeros(
            (1, self.n_features),
        )  # overwritten at init_background
        self.init_background(self.data)

        if normalize:  # update normalization value
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Imputes the missing values of a data point and calls the model.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which
                are missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
               ``(n_subsets, n_outputs)``.

        """
        n_coalitions = coalitions.shape[0]
        replacement_data = self._sample_replacement_data(self.sample_size)
        sample_size = replacement_data.shape[0]
        outputs = np.zeros((sample_size, n_coalitions))
        imputed_data = np.tile(np.copy(self._x), (n_coalitions, 1))
        for j in range(sample_size):
            for i in range(n_coalitions):
                imputed_data[i, ~coalitions[i]] = replacement_data[j, ~coalitions[i]]
            predictions = self.predict(imputed_data)
            outputs[j] = predictions
        outputs = np.mean(outputs, axis=0)  # average over the samples
        # insert the better approximate empty prediction for the empty coalitions
        outputs[~np.any(coalitions, axis=1)] = self.empty_prediction
        return outputs

    def init_background(self, data: np.ndarray) -> MarginalImputer:
        """Initializes the imputer to the background data.

        The background data is used to sample replacement values for the missing features.
        To change the background data, use this method.

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

        """
        self.replacement_data = np.copy(data)
        if self.sample_size > self.replacement_data.shape[0]:
            warnings.warn(UserWarning(_too_large_sample_size_warning), stacklevel=2)
            self.sample_size = self.replacement_data.shape[0]
        self.calc_empty_prediction()  # reset the empty prediction to the new background data
        return self

    def _sample_replacement_data(self, sample_size: int | None = None) -> np.ndarray:
        """Samples replacement values from the background data.

        Args:
            sample_size: The number of replacement values to sample. If ``None``, all replacement
                values are sampled. Defaults to ``None``.

        Returns:
            The replacement values as a two-dimensional array with shape
                ``(sample_size, n_features)``.

        """
        replacement_data = np.copy(self.replacement_data)
        rng = np.random.default_rng(self.random_state)
        if not self.joint_marginal_distribution:  # shuffle data to break joint marginal dist.
            for feature in range(self.n_features):
                rng.shuffle(replacement_data[:, feature])
        n_samples = replacement_data.shape[0]
        if sample_size is None or sample_size >= n_samples:
            return replacement_data
        # sample replacement values
        replacement_idx = rng.choice(n_samples, size=sample_size, replace=False)
        return replacement_data[replacement_idx]

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.

        """
        background_data = self._sample_replacement_data()
        empty_predictions = self.predict(background_data)
        empty_prediction = float(np.mean(empty_predictions))
        self.empty_prediction = empty_prediction
        if self.normalize:  # reset the normalization value
            self.normalization_value = empty_prediction
        return empty_prediction
