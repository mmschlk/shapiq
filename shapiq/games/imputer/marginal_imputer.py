"""Implementation of the marginal imputer."""

from typing import Callable, Optional

import numpy as np

from shapiq.games.imputer.base import Imputer


class MarginalImputer(Imputer):
    """The marginal imputer for the shapiq package.

    The marginal imputer is used to impute the missing values of a data point by using the
    marginal distribution of the background data.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        data: The background data to use for the explainer as a two-dimensional array
            with shape ``(n_samples, n_features)``.
        x: The explanation point to use the imputer to.
        sample_replacements: Whether to sample replacements from the background data or to use the
            mean (for numerical features) or the median (for categorical features) of the background
            data. Defaults to ``True``.
        sample_size: The number of samples to draw from the background data. Only used if
            ``sample_replacements`` is ``True``. Increasing this value will linearly increase the
            runtime of the explainer. Defaults to ``100``.
        categorical_features: A list of indices of the categorical features in the background data.
            If no categorical features are given, all features are assumed to be numerical or in
            string format (where ``np.mean`` fails) features. Defaults to ``None``.
        normalize: A flag to normalize the game values. If ``True``, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to ``True``.
        random_state: The random state to use for sampling. Defaults to ``None``.

    Attributes:
        replacement_data: The data to use for imputation. Either samples from the background data
            or the mean / median of the background data.
        empty_prediction: The model's prediction on an empty data point (all features missing).
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        x: Optional[np.ndarray] = None,
        sample_replacements: bool = True,
        sample_size: int = 100,
        categorical_features: list[int] = None,
        normalize: bool = True,
        random_state: Optional[int] = None,
        joint_marginal_distribution: bool = True,  # TODO: changed from False
        cond_sampler: Optional[Callable] = None,
        take_all_background: bool = False,
    ) -> None:
        super().__init__(model, data, categorical_features, random_state)

        # setup attributes
        self._sample_replacements = sample_replacements
        self._sample_size: int = sample_size
        self.replacement_data: np.ndarray = np.zeros((1, self._n_features))  # will be overwritten
        self.take_all_background: bool = take_all_background
        self.init_background(self.data)
        self._x: np.ndarray = np.zeros((1, self._n_features))  # will be overwritten @ fit
        if x is not None:
            self.fit(x)

        self.joint_marginal_distribution: bool = joint_marginal_distribution
        self.replacement_sampler: Optional[Callable] = cond_sampler

        # set empty value and normalization
        self.empty_prediction: float = self._calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray[float]:
        """Imputes the missing values of a data point and calls the model.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which are
                missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
               ``(n_subsets, n_outputs)``.
        """
        n_coalitions = coalitions.shape[0]
        data = np.tile(np.copy(self._x), (n_coalitions, 1))
        if self._sample_replacements:
            # sampling from background returning array of shape (sample_size, n_subsets, n_features)
            if not self.take_all_background:
                if self.replacement_sampler is not None:
                    replacement_data = self.replacement_sampler(coalitions, x_to_impute=self._x)
                else:
                    replacement_data = self._sample_replacement_values(coalitions)
                outputs = np.zeros((self._sample_size, n_coalitions))
                for i in range(self._sample_size):
                    replacements = replacement_data[i].reshape(n_coalitions, self._n_features)
                    data[~coalitions] = replacements[~coalitions]
                    outputs[i] = self.predict(data)
            else:
                outputs = np.zeros((self.replacement_data.shape[0], n_coalitions))
                for i in range(self.replacement_data.shape[0]):
                    data[~coalitions] = np.tile(self.replacement_data[i], (n_coalitions, 1))[
                        ~coalitions
                    ]
                    outputs[i] = self.predict(data)
            outputs = np.mean(outputs, axis=0)  # average over the samples
        else:
            replacement_data = np.tile(self.replacement_data, (n_coalitions, 1))
            data[~coalitions] = replacement_data[~coalitions]
            outputs = self.predict(data)
        return outputs

    def init_background(self, data: np.ndarray) -> "MarginalImputer":
        """Initializes the imputer to the background data.

        Args:
            data: The background data to use for the imputer. The shape of the array must
                be ``(n_samples, n_features)``.

        Returns:
            The initialized imputer.
        """
        if self._sample_replacements:
            if self.take_all_background and self.data.shape[0] < self._sample_size:
                # if background data is smaller then increase it by repeating it until it is sample_size
                missing_rows = self._sample_size - self.data.shape[0]
                sample_indices = np.random.choice(self.data.shape[0], missing_rows, replace=True)
                data = np.concatenate((data, self.data[sample_indices]))
            self.replacement_data = data
        else:
            self.replacement_data = np.zeros((1, self._n_features), dtype=object)
            for feature in range(self._n_features):
                feature_column = data[:, feature]
                if feature in self._cat_features:
                    # get mode for categorical features
                    counts = np.unique(feature_column, return_counts=True)
                    summarized_feature = counts[0][np.argmax(counts[1])]
                else:
                    try:  # try to use mean for numerical features
                        summarized_feature = np.mean(feature_column)
                    except TypeError:  # fallback to mode for string features
                        counts = np.unique(feature_column, return_counts=True)
                        summarized_feature = counts[0][np.argmax(counts[1])]
                self.replacement_data[:, feature] = summarized_feature
        return self

    def fit(self, x: np.ndarray[float]) -> "MarginalImputer":
        """Fits the imputer to the explanation point.

        Args:
            x: The explanation point to use the imputer to.

        Returns:
            The fitted imputer.
        """
        self._x = x
        return self

    def _sample_replacement_values(self, coalitions: np.ndarray[bool]) -> np.ndarray:
        """Samples replacement values from the background data.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which are
                missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The sampled replacement values. The shape of the array is ``(sample_size, n_subsets,
                n_features)``.
        """
        n_coalitions = coalitions.shape[0]
        replacement_data = np.zeros(
            (self._sample_size, n_coalitions, self._n_features), dtype=object
        )
        if not self.joint_marginal_distribution:
            for feature in range(self._n_features):
                sampled_feature_values = self._rng.choice(
                    self.replacement_data[:, feature],
                    size=(self._sample_size, n_coalitions),
                    replace=True,
                )
                replacement_data[:, :, feature] = sampled_feature_values
        else:
            for i in range(n_coalitions):
                _rng = np.random.default_rng(self._random_state)
                sampled_indices = _rng.choice(
                    self.replacement_data.shape[0],
                    size=self._sample_size,
                    replace=True,
                )
                replacement_data[:, i, :] = self.replacement_data[sampled_indices]
        return replacement_data

    def _calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        empty_predictions = self.predict(self.replacement_data)
        if self._sample_replacements:
            empty_prediction = np.mean(empty_predictions)
        else:
            empty_prediction = empty_predictions[0]
        return empty_prediction
