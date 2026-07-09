"""Tests for teaching errors and validation hardening across entry points."""

from __future__ import annotations

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
    explanation = ExactExplainer(additive_game(), SV).explain()
    assert jnp.allclose(explanation((np.int64(0),)), explanation((0,)), atol=0)


def test_wrong_player_types_and_ranges_are_named():
    explanation = ExactExplainer(additive_game(), SV).explain()
    with pytest.raises(TypeError, match="got str"):
        explanation(("a",))
    with pytest.raises(ValueError, match="out of range for 5 players"):
        explanation((7,))


def test_index_construction_is_rejected_with_teaching_errors():
    with pytest.raises(TypeError, match="already the index"):
        SII(order=2)
    with pytest.raises(TypeError, match="already the index"):
        SV()


def test_index_classes_are_rejected_with_teaching_errors():
    with pytest.raises(TypeError, match="pass the _ShortWeights value itself"):
        ExactExplainer(additive_game(), _ShortWeights)


def test_non_conforming_indices_name_the_missing_members():
    class _OldStyle:
        name = "SII"

    with pytest.raises(TypeError, match="missing index members: order_semantics"):
        ExactExplainer(additive_game(), _OldStyle())


def test_missing_order_teaches_the_explainer_grammar():
    with pytest.raises(TypeError, match="pass order= to the explainer"):
        ExactExplainer(additive_game(), SII)
    with pytest.raises(TypeError, match="pass order= to the explainer"):
        Regression(additive_game(), FSII)
    with pytest.raises(TypeError, match="pass order= to the explainer"):
        PermutationSampling(additive_game(), SII)


def test_string_indices_teach_the_class_grammar_everywhere():
    with pytest.raises(TypeError, match=r"not the string 'FSII'"):
        Regression(additive_game(), "FSII")
    with pytest.raises(TypeError, match=r"not the string 'SV'"):
        PermutationSampling(additive_game(), "SV")


class _ShortWeights:
    """A cardinal index whose weight vector is one element too short."""

    name = "SII"  # borrow a shipped name to isolate the length check
    order_semantics = "coverage"
    includes_empty_interaction = False
    min_interaction_size = 1
    preserves_value = True
    generalizes = None

    def resolve_order(self, order, *, n_players):
        del n_players
        return 1 if order is None else order

    def derivative_weights(self, n_players, interaction_size, *, order):
        del order
        return jnp.ones(n_players - interaction_size)  # one short of n - s + 1


def test_short_weight_vectors_are_rejected_instead_of_clamped():
    explainer = ExactExplainer(additive_game(), _ShortWeights())
    with pytest.raises(ValueError, match="must return 5 weights"):
        explainer.explain()


def test_identification_hint_adapts_to_deduplication():
    game = additive_game()
    with pytest.raises(InsufficientSamplesError, match="deduplicate=True"):
        Regression(game, FSII, order=2, random_state=0).sample(2 + 6).explain()
    with pytest.raises(InsufficientSamplesError, match="retry") as caught:
        Regression(game, FSII, order=2, random_state=0, deduplicate=True).sample(2 + 6).explain()
    assert "deduplicate=True" not in str(caught.value)
