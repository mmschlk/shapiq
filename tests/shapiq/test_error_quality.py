"""Tests for teaching errors and validation hardening across entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    FSII,
    SII,
    SV,
    CallableGame,
    ExactExplainer,
    InsufficientSamplesError,
    PermutationSampling,
    Regression,
)

N_PLAYERS = 5
WEIGHTS = jnp.asarray([0.7, -1.3, 0.1, 2.0, -0.4])


def additive_game():
    return CallableGame(
        fn=lambda c: jnp.asarray(c.to_dense(), dtype=jnp.float32) @ WEIGHTS,
        n_players=N_PLAYERS,
    )


def test_numpy_integer_player_indices_are_accepted():
    explanation = ExactExplainer(additive_game(), SV()).explain()
    assert jnp.allclose(explanation((np.int64(0),)), explanation((0,)), atol=0)


def test_wrong_player_types_and_ranges_are_named():
    explanation = ExactExplainer(additive_game(), SV()).explain()
    with pytest.raises(TypeError, match="got str"):
        explanation(("a",))
    with pytest.raises(ValueError, match="out of range for 5 players"):
        explanation((7,))


def test_index_classes_are_rejected_with_teaching_errors():
    with pytest.raises(TypeError, match="not the SII class"):
        ExactExplainer(additive_game(), SII)
    with pytest.raises(TypeError, match="not the FSII class"):
        Regression(additive_game(), FSII)
    with pytest.raises(TypeError, match="not the SV class"):
        PermutationSampling(additive_game(), SV)


def test_string_indices_teach_the_object_grammar_everywhere():
    with pytest.raises(TypeError, match=r"instead of 'FSII'"):
        Regression(additive_game(), "FSII")
    with pytest.raises(TypeError, match=r"instead of 'SV'"):
        PermutationSampling(additive_game(), "SV")


@dataclass(frozen=True)
class _ShortWeights:
    """A cardinal index whose weight vector is one element too short."""

    order: int = 1

    name: ClassVar = "SII"  # borrow a shipped name to isolate the length check
    order_semantics: ClassVar = "coverage"
    orientation: ClassVar = "undirected"
    includes_empty_interaction: ClassVar = False
    min_interaction_size: ClassVar = 1
    preserves_value: ClassVar = True
    generalizes: ClassVar = None

    def derivative_weights(self, n_players: int, interaction_size: int):
        return jnp.ones(n_players - interaction_size)  # one short of n - s + 1


def test_short_weight_vectors_are_rejected_instead_of_clamped():
    explainer = ExactExplainer(additive_game(), _ShortWeights())
    with pytest.raises(ValueError, match="must return 5 weights"):
        explainer.explain()


def test_identification_hint_adapts_to_deduplication():
    game = additive_game()
    with pytest.raises(InsufficientSamplesError, match="deduplicate=True"):
        Regression(game, FSII(order=2), random_state=0).sample(2 + 6).explain()
    with pytest.raises(InsufficientSamplesError, match="retry") as caught:
        Regression(game, FSII(order=2), random_state=0, deduplicate=True).sample(2 + 6).explain()
    assert "deduplicate=True" not in str(caught.value)
