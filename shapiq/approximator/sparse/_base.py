from collections.abc import Callable
from .._base import Approximator
from ...interaction_values import InteractionValues
from ...game_theory.moebius_converter import MoebiusConverter
from sparse_transform.qsft.qsft import transform
from sparse_transform.qsft.codes.BCH import BCH
from sparse_transform.qsft.signals.input_signal_subsampled import SubsampledSignal
from functools import partial
import numpy as np

class Sparse(Approximator):

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        top_order: bool,
        random_state: int | None = None,
        transform_type: str = "fourier", # TODO JK: New parameter, to implement fourier or mobius transform?
    ) -> None:
        self.transform_type = transform_type
        super().__init__(
            n=n, max_order=max_order, index=index, top_order=top_order, random_state=random_state,
        )

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> InteractionValues:
        """Approximates the interaction values using a sparse transform approach.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.

        Returns:
            The approximated Shapley interaction values.
        """
        b = self._compute_b(budget, self.transform_type)
        signal = self._sample_fourier(game, b)
        transform = Sparse._support_recovery_fourier(signal, b)
        return self._converter(transform)

    def _compute_b(self, budget: int, transform_type: str) -> int:
        """Computes the budget for the approximation.

        Args:
            budget: The user defined budget for the approximation.

        Returns:
            The actual b availible for computing the transform.
        """
        #TODO JK: Compute the reduction in budget given the user defined upper bound
        b = 6
        return b

    def _sample_fourier(self, game, b, t=5):
        query_args = {
            "query_method": "complex",
            "num_subsample": 3,
            "delays_method_source": "joint-coded",
            "subsampling_method": "qsft",
            "delays_method_channel": "identity-siso",
            "num_repeat": 1,
            "b": b,
            "t": t
        }
        signal = SubsampledSignal(func=game, n=self.n, q=2, query_args=query_args)
        return signal

    def _converter(self, transform):
        temp_mobius = Sparse._fourier_to_mobius(transform)
        autoconverter = MoebiusConverter(moebius_coefficients=temp_mobius)
        return autoconverter(index=self.index, order=self.max_order)

    @staticmethod
    def _fourier_to_mobius(transform):
        x = InteractionValues()
        return x
        # TODO JK: We need to implement this

    @staticmethod
    def _support_recovery_fourier(signal, b, t=5, type="soft"):
        if type == "hard":
            source_decoder = BCH(signal.n, t).parity_decode
        else:
            source_decoder = partial(BCH(signal.n, t).parity_decode_2chase_t2_max_likelihood,
                                     chase_depth=2 * t)
        qsft_args = {
            "num_subsample": 3,
            "num_repeat": 1,
            "reconstruct_method_source": "coded",
            "reconstruct_method_channel": "identity-siso" if type != "hard" else "identity",
            "b": b,
            "source_decoder": source_decoder,
            "peeling_method": "multi-detect",
            "noise_sd": 0,
            "regress": 'lasso',
            "res_energy_cutoff": 0.9,
            "trap_exit": True,
            "verbosity": 0,
            "report": False,
            "peel_average": True,
        }
        return {key: np.real(value) for key, value in transform(signal, **qsft_args).items()}