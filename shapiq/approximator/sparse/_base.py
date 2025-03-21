from collections.abc import Callable
from .._base import Approximator
from ...interaction_values import InteractionValues
from ...game_theory.moebius_converter import MoebiusConverter
from sparse_transform.qsft.qsft import sparse_fourier_transform
from sparse_transform.qsft.codes.BCH import BCH
from sparse_transform.qsft.signals.input_signal_subsampled import SubsampledSignalFourier
from sparse_transform.smt.smt import sparse_moebius_transform
from sparse_transform.smt.signals.input_signal_subsampled import SubsampledSignalMoebius
from ..utils.sets import powerset
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
        transform_type: str = "fourier",
    ) -> None:
        assert transform_type.lower() in ["fourier", "moebius"], "transform_type must be either 'fourier' or 'mobius'"
        self.transform_type = transform_type.lower()
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
        b, used_budget = self._compute_b(budget, self.transform_type)

        if self.transform_type == "fourier":
            signal = self._sample_fourier(game, b)
            fourier_transform = Sparse._support_recovery_fourier(signal, b, t=max(5, self.max_order))
            moebius_transform = Sparse._fourier_to_moebius(fourier_transform, t=max(5, self.max_order))
        else:
            signal = self._sample_moebius(game, b)
            moebius_transform = Sparse._support_recovery_moebius(signal, b)

        moebius_interactions = list(moebius_transform.keys())
        self._interaction_lookup = {i: key for i, key in enumerate(moebius_interactions)}
        values = np.array([moebius_transform[key] for key in moebius_interactions])
        mobius_interactions = InteractionValues(
            values=values,
            index="Moebius",
            min_order=min(moebius_interactions, key=len),
            max_order=max(moebius_interactions, key=len),
            n_players=self.n,
            interaction_lookup=self._interaction_lookup,
            estimated=True,
            estimation_budget=used_budget,
            baseline_value=moebius_interactions[tuple()] if tuple() in moebius_interactions else 0.0,
        )

        autoconverter = MoebiusConverter(moebius_coefficients=mobius_interactions)
        return autoconverter(index=self.index, order=self.max_order)



    def _compute_b(self, budget: int) -> int:
        """Computes the budget for the approximation.

        Args:
            budget: The user defined budget for the approximation.

        Returns:
            The actual b availible for computing the transform.
            The sample budget to be used under this b.
        """
        #TODO JK: Compute the reduction in budget given the user defined upper bound
        b = 6
        used_budget = 0
        return b, used_budget

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
        signal = SubsampledSignalFourier(func=game, n=self.n, q=2, query_args=query_args)
        return signal

    def _sample_moebius(self, game, b):
        query_args = {
            "query_method": "simple",
            "num_subsample": 3,
            "delays_method_source": "identity",
            "delays_method_channel": "identity",
            "subsampling_method": "smt",
            "num_repeat": 1,
            "b": b,
        }
        signal = SubsampledSignalMoebius(func=game, n=self.n, q=2, query_args=query_args)
        return signal

    @staticmethod
    def _fourier_to_moebius(fourier_transform):
        moebius_dict = {}
        for loc, coef in fourier_transform.items():
            for subset in powerset(loc):
                scaling_factor = np.power(-2.0, len(subset))
                if subset in moebius_dict:
                    moebius_dict[subset] += coef * scaling_factor
                else:
                    moebius_dict[subset] = coef * scaling_factor
        return moebius_dict

    @staticmethod
    def _support_recovery_fourier(signal, b, t=5, type="soft"):

        if type == "hard":
            source_decoder = BCH(signal.n, t).parity_decode
        else:
            source_decoder = partial(BCH(signal.n, t).parity_decode_2chase_t2_max_likelihood,
                                     chase_depth=2 * t)
        spex_args = {
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

        return {tuple(np.nonzero(key)[0]): np.real(value) for key, value in
                            sparse_fourier_transform(signal, **spex_args).items()}

    @staticmethod
    def _support_recovery_moebius(signal, b):
        smt_args = {
            "num_subsample": 3,
            "num_repeat": 1,
            "reconstruct_method_source": "simple",
            "reconstruct_method_channel": "simple",
            "b": b,
            "noise_sd": 0,
            "source_decoder": None
        }
        return {tuple(np.nonzero(key)[0]): np.real(value) for key, value in
                            sparse_moebius_transform(signal, **smt_args).items()}

    #####################################################################
    # These functions will be replaced in the next release of sparse-transform
    #####################################################################

    @staticmethod
    def get_number_of_samples(n, b, t, q, query_args):
        """
        Computes the number of vector-wise calls to self.func for the given query_args, n, t, and b.
        """
        num_subsample = query_args.get("num_subsample", 1)
        num_rows_per_D = SubsampledSignal._get_delay_overhead(n, t, query_args)
        samples_per_row = q ** b
        total_samples = num_subsample * num_rows_per_D * samples_per_row  # upper bound
        return total_samples

    @staticmethod
    def get_b_for_sample_budget(budget, n, t, q, query_args):
        """
        Find the maximum value of b that fits within the given sample budget.

        Parameters:
        budget (int): The maximum number of samples allowed.
        n (int): Number of rows.
        t (int): Error parameter.
        q (int): Base of the transform.
        query_args (dict): Additional query arguments.

        Returns:
        int: The maximum value of b that keeps the total samples within budget.
        """
        num_subsample = query_args.get("num_subsample", 1)
        num_rows_per_D = SubsampledSignal._get_delay_overhead(n, t, query_args)
        largest_b = np.floor(np.log(budget / (num_rows_per_D * num_subsample)) / np.log(q))
        return int(largest_b)

    @staticmethod
    def _get_delay_overhead(n, t, query_args):  # TODO depends on q in general
        """
        Returns the overhead of the delays in terms of the number of samples
        """
        delays_method_source = query_args.get("delays_method_source", "identity")
        if delays_method_source == "identity":
            num_rows_per_D = n + 1
        elif delays_method_source == "joint-coded":
            from sparse_transform.qsft.codes.BCH import BCH
            nc, kc = BCH.parameter_search(n, t)
            num_rows_per_D = nc - kc + 1  # BCH parity length + 1 (for zero row)
        elif delays_method_source == "random":
            # For random delays, the number is specified or defaulted
            num_rows_per_D = query_args.get("num_delays", n)
        else:
            # For other delay methods, assume default behavior
            num_rows_per_D = n + 1

        if query_args.get("delays_method_channel") == "nso":
            num_repeat = query_args.get("num_repeat", 1)
        else:
            num_repeat = 1
        return num_rows_per_D * num_repeat