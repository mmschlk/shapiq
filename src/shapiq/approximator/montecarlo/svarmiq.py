"""SVARM-IQ approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

from .base import MonteCarlo, ValidMonteCarloIndices

if TYPE_CHECKING:
    from shapiq.typing import FloatVector

TIndices = TypeVar("TIndices", bound=ValidMonteCarloIndices)
"""A type variable for the valid indices of the MonteCarlo approximator."""


class SVARMIQ(MonteCarlo[ValidMonteCarloIndices]):
    """The SVARM-IQ approximator for Shapley interactions.

    SVARM-IQ utilizes MonteCarlo approximation with two stratification strategies. SVARM-IQ is a
    generalization of the SVARM algorithm :cite:p:`Kolpaczki.2024a` and can approximate
    any-order Shapley interactions efficiently. For details about the algorithm see the original
    paper by :cite:t:`Kolpaczki.2024b`.

    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: ValidMonteCarloIndices = "k-SII",
        *,
        top_order: bool = False,
        pairing_trick: bool = False,
        sampling_weights: FloatVector | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the SVARMIQ approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            index: The interaction index to be used. Choose from ``['k-SII', 'SII']``. Defaults to
                ``'k-SII'``.

            top_order: If ``True``, the top-order interactions are estimated. Defaults to ``False``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            random_state: The random state of the estimator. Defaults to ``None``.

        """
        super().__init__(
            n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            stratify_coalition_size=True,
            stratify_intersection=True,
            random_state=random_state,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
        )


ValidIndicesSVARM = Literal["SV", "BV"]


class SVARM(MonteCarlo[ValidIndicesSVARM]):
    """The SVARM approximator for estimating the Shapley value (SV).

    SVARM is a MonteCarlo approximation algorithm that estimates the Shapley value. For details
    about the algorithm see the original paper by Kolpaczki et al. (2024)
    :footcite:t:`Kolpaczki.2024a`.

    References:
        .. footbibliography::

    """

    valid_indices: tuple[ValidIndicesSVARM, ...] = tuple(get_args(ValidIndicesSVARM))
    """The valid indices for the SVARM approximator."""

    def __init__(
        self,
        n: int,
        index: ValidIndicesSVARM = "SV",
        *,
        random_state: int | None = None,
        pairing_trick: bool = False,
        sampling_weights: FloatVector | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the SVARM approximator.

        Args:
            n: The number of players.

            index: The interaction index to be used. Choose from ``['SV', 'BV']``. Defaults to
                ``'SV'``.

            random_state: The random state of the estimator. Defaults to ``None``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            **kwargs: Additional keyword arguments (not used only for compatibility).
        """
        super().__init__(
            n,
            max_order=1,
            index=index,
            top_order=False,
            stratify_coalition_size=True,
            stratify_intersection=True,
            random_state=random_state,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
        )
