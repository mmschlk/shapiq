"""This test module contains all tests regarding the base sparse approximator."""

from __future__ import annotations

import pytest

from shapiq.approximator.sparse import Sparse
from shapiq.game_theory.indices import ALL_AVAILABLE_CONCEPTS
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize(
    "n, index, max_order, top_order, transform_type, decoder_type",
    [
        (7, "FBII", None, False, "fourier", "soft"),
        (7, "FBII", None, False, "FouRier", "Soft"),
        (7, "FBII", None, False, "fourier", "Hard"),
        (7, "wrong_index", None, False, "fourier", "soft"),  # Should raise ValueError
        (7, "wrong_index", None, False, "fourier", "wrong_dec_type"),  # Should raise ValueError
        (7, "FBII", None, False, "invalid_type", None),  # Should raise ValueError
        (7, "FBII", 6, False, "fourier", "soft"),
        (7, "FBII", 2, False, "fourier", "soft"),
    ],
)
def test_initialization(n, index, max_order, top_order, transform_type, decoder_type):
    """Tests the initialization of the Sparse approximator."""

    if index == "wrong_index":
        with pytest.raises(ValueError):
            _ = Sparse(
                n=n,
                index=index,
                max_order=max_order,
                top_order=top_order,
                transform_type=transform_type,
                decoder_type=decoder_type,
            )
        return

    if transform_type == "invalid_type":
        with pytest.raises(ValueError):
            _ = Sparse(
                n=n,
                index=index,
                max_order=max_order,
                top_order=top_order,
                transform_type=transform_type,
                decoder_type=decoder_type,
            )
        return

    if decoder_type == "wrong_dec_type" or (
        transform_type.lower() == "mobius" and decoder_type is not None
    ):
        with pytest.raises(ValueError):
            _ = Sparse(
                n=n,
                index=index,
                max_order=max_order,
                top_order=top_order,
                transform_type=transform_type,
                decoder_type=decoder_type,
            )
        return

    approximator = Sparse(
        n=n,
        index=index,
        max_order=max_order,
        top_order=top_order,
        transform_type=transform_type,
        decoder_type=decoder_type,
    )

    assert approximator.n == n
    assert approximator.max_order == (n if max_order is None else max_order)
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 0)
    assert approximator.index == index
    assert approximator.transform_type == transform_type.lower()
    if decoder_type is not None or transform_type.lower() == "fourier":
        channel_method = approximator.decoder_args["reconstruct_method_channel"]
        if channel_method is None or decoder_type.lower() == "soft":
            assert channel_method == "identity-siso"
        else:
            assert channel_method == "identity"


@pytest.mark.parametrize(
    "n, index, max_order, budget, transform_type, decoder_type, top_order, interaction",
    [  # These test usually pass with much fewer samples, (~90% of the time with 800), but we use a bigger budget to
        # ensure that the tests are stable
        (20, "STII", None, 1600, "fourier", "soft", False, (1, 2)),  # Standard configuration (STII)
        (
            20,
            "FBII",
            None,
            6400,
            "fourier",
            "hard",
            False,
            (1, 2, 4, 5),
        ),  # Higher order interaction
        (20, "FSII", None, 1600, "fourier", "soft", False, (0, 2)),  # Standard configuration (FSII)
        (
            20,
            "STII",
            None,
            100,
            "fourier",
            "soft",
            False,
            (1, 2),
        ),  # Should throw and error budget too small
        (
            20,
            "STII",
            2,
            1600,
            "fourier",
            "soft",
            True,
            (1, 2),
        ),  # Should filter out all 1st order interactions
        (
            20,
            "STII",
            3,
            3200,
            "fourier",
            "soft",
            False,
            (1, 2, 3),
        ),  # Should filter out 3rd order interaction
        (
            20,
            "STII",
            1,
            3200,
            "fourier",
            "soft",
            False,
            (1, 2, 3),
        ),  # Should spread 3rd order interaction across 1st
    ],
)
def test_approximate(
    n, index, max_order, budget, transform_type, decoder_type, top_order, interaction
):
    """Tests the approximation of the Sparse approximator with various configurations."""
    game = DummyGame(n, interaction)
    approximator = Sparse(
        n=n,
        index=index,
        max_order=max_order,
        top_order=top_order,
        transform_type=transform_type,
        decoder_type=decoder_type,
        random_state=42,
    )
    # Perform approximation with budget
    if n >= 20 and budget <= 100:
        with pytest.raises(ValueError):
            _ = approximator.approximate(budget, game)
        return

    estimates = approximator.approximate(budget, game)
    max_order = n if max_order is None else max_order

    # Verify the result structure
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 0)
    assert (
        estimates.index == index
        if (max_order is None or max_order > 1)
        else ALL_AVAILABLE_CONCEPTS[index]["generalizes"]
    )
    assert estimates.estimated
    assert estimates.estimation_budget > 0

    # Check that game was called within budget
    assert game.access_counter <= budget + 2

    # Check that values are not empty
    assert len(estimates.values) > 0

    # generate the set of expected interactions
    expected_interactions = set()
    if estimates.min_order == 0:
        expected_interactions.update({(i,) for i in range(n)})
    if estimates.max_order > 1:
        expected_interactions.add(interaction)

    # Check the computed interactions
    recovered_interactions = set(estimates.interaction_lookup.keys())
    zero_interactions = set()
    if index == "STII" or index == "FBII" or index == "FSII":
        for interaction_key in recovered_interactions:
            if interaction_key not in expected_interactions:
                assert estimates[interaction_key] == pytest.approx(0.0, abs=1e-2)
                zero_interactions.add(interaction_key)
            else:
                if interaction_key == interaction and max_order >= len(interaction):
                    assert estimates[interaction_key] == pytest.approx(1.0, rel=0.01)
                else:
                    if (
                        len(interaction_key) == 1
                        and interaction_key[0] in interaction
                        and max_order == 1
                    ):
                        assert estimates[interaction_key] == pytest.approx(
                            1 / n + 1 / len(interaction), rel=0.01
                        )
                    else:
                        assert estimates[interaction_key] == pytest.approx(1 / n, rel=0.01)
    recovered_interactions = recovered_interactions - zero_interactions
    assert recovered_interactions == expected_interactions
