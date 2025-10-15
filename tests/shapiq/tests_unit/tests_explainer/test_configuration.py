"""Tests the configuration module for the explainer."""

from __future__ import annotations

from typing import get_args

import pytest

from shapiq.approximator import (
    SVARMIQ,
    KernelSHAP,
    KernelSHAPIQ,
    RegressionFBII,
    RegressionFSII,
)
from shapiq.explainer.configuration import ValidApproximatorTypes


class TestAutomaticSelection:
    """Tests the automatic selection of approximators based on indices and orders."""

    def test_choosing_of_spex(self):
        """Tests if SPEX is chosen appropriately for large max_order."""
        from shapiq.explainer.configuration import choose_spex

        n_players = list(range(101))  # from 0 to 100 players
        orders = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        for order in orders:
            for n in n_players:
                use_spex = choose_spex(n_players=n, max_order=order)
                # check if SPEX is chosen correctly
                if order <= 1 or n <= 16:
                    assert not use_spex
                elif (
                    (order > 1 and n > 64)
                    or (order == 3 and n > 32)
                    or (order == 4 and n > 16)
                    or (order > 4 and n > 16)
                ):
                    assert use_spex
                else:
                    assert not use_spex, (
                        f"Unexpected SPEX choice for n_players={n}, max_order={order}."
                    )

    @pytest.mark.parametrize(
        "index, order, n_players, expected_approx",
        [
            # SV correctly uses KernelSHAP
            ("SV", 1, 10, KernelSHAP),
            ("SV", 1, 101, KernelSHAP),
            ("SII", 1, 10, KernelSHAP),
            ("k-SII", 1, 10, KernelSHAP),
            ("FSII", 1, 10, KernelSHAP),
            ("STII", 1, 10, KernelSHAP),
            ("SV", 2, 10, KernelSHAP),  # SV is selected even if max_order > 1
            # BV correctly uses RegressionFBII
            ("BV", 1, 10, RegressionFBII),
            ("BV", 1, 101, RegressionFBII),
            ("BV", 2, 10, RegressionFBII),  # BV is selected even if max_order > 1
            # RegressionFSII is selected correctly
            ("FSII", 2, 10, RegressionFSII),
            ("FSII", 3, 10, RegressionFSII),
            # RegressionFBII is selected correctly
            ("FBII", 2, 10, RegressionFBII),
            ("FBII", 3, 10, RegressionFBII),
            # KernelSHAPIQ is selected correctly
            ("SII", 2, 10, KernelSHAPIQ),
            ("k-SII", 2, 10, KernelSHAPIQ),
            ("SII", 3, 10, KernelSHAPIQ),
            ("k-SII", 3, 10, KernelSHAPIQ),
            # we omit SPEX since it takes a longer time
            # test SVARMIQ
            ("STII", 2, 10, SVARMIQ),
            ("STII", 3, 10, SVARMIQ),
        ],
    )
    def test_setup_approximator_automatically(self, index, order, n_players, expected_approx):
        """Checks if the automatic setup of the approximator works correctly."""
        from shapiq.explainer.configuration import setup_approximator_automatically

        approx = setup_approximator_automatically(index=index, max_order=order, n_players=n_players)
        assert isinstance(approx, expected_approx)


class TestSetupApproximator:
    """Tests the setup of the approximator with different configurations."""

    def test_setup_approximator_with_existing_approx(self):
        """Tests if the setup of the approximator with an existing approximator returns the instance."""
        from shapiq.explainer.configuration import setup_approximator

        approx = KernelSHAP(n=10)
        approx_r = setup_approximator(approximator=approx, index="SV", max_order=1, n_players=10)
        assert approx_r is approx, (
            "The returned approximator should be the same as the input approximator."
        )

    def test_setup_approximator_with_auto(self):
        """Tests if the setup of the approximator with 'auto' returns an approximator instance."""
        from shapiq.approximator.base import Approximator
        from shapiq.explainer.configuration import setup_approximator

        approx = setup_approximator(approximator="auto", index="SV", max_order=1, n_players=10)
        assert isinstance(approx, Approximator)

    @pytest.mark.parametrize("approx_name", list(get_args(ValidApproximatorTypes)))
    def test_setup_approximator_from_string(self, approx_name: ValidApproximatorTypes):
        """Tests if the setup of the approximator from a string returns an instance of the correct type."""
        from shapiq.explainer.configuration import APPROXIMATOR_CONFIGURATIONS, setup_approximator

        available_indices = APPROXIMATOR_CONFIGURATIONS[approx_name].keys()
        for index in available_indices:
            order = 1 if index in ["SV", "BV"] else 2
            approx_class = APPROXIMATOR_CONFIGURATIONS[approx_name][index]
            approx = setup_approximator(
                approximator=approx_name, index=index, max_order=order, n_players=10
            )
            assert isinstance(approx, approx_class), (
                f"Expected {approx_class} for {approx_name} with index {index}, "
                f"but got {type(approx_class)}."
            )


class TestErrorCases:
    """Tests for error cases for the setup_approximator function."""

    def test_setup_approximator_with_invalid_approximator(self):
        """Tests that an error is raised when an invalid approximator is passed."""
        from shapiq.explainer.configuration import setup_approximator

        with pytest.raises(ValueError, match="Invalid approximator `invalid`."):
            setup_approximator(approximator="invalid", index="SV", max_order=1, n_players=10)

    def test_setup_approximator_with_invalid_class_as_approx(self):
        """Tests that an error is raised when an invalid class is passed as approximator."""
        from shapiq.explainer.configuration import setup_approximator

        class InvalidApproximator:
            pass

        with pytest.raises(TypeError, match="Invalid approximator "):
            setup_approximator(
                approximator=InvalidApproximator, index="SV", max_order=1, n_players=10
            )
