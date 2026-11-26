"""Dtype inference for traceable and opaque models without concrete predictions."""

from dataclasses import dataclass, replace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game import CastGame
from shapiq.masker import BaselineMasker, MarginalMasker
from shapiq.players import FinitePlayers
from shapiq.value import VectorValue
from shapiq.xai import MaskedGame, infer_model_spec


def test_jax_model_spec_traces_one_row_for_scalar_and_vector_values():
    traced_shapes = []

    def vector_model(x):
        assert isinstance(x, jax.core.Tracer)
        traced_shapes.append(x.shape)
        return x[:, 0, :].astype(jnp.bfloat16)

    def weak_model(x):
        return jnp.full((x.shape[0],), 3.0)

    sample = jnp.ones((6, 2, 3))
    result = infer_model_spec(vector_model, sample)
    assert result == jax.ShapeDtypeStruct((), jnp.bfloat16)
    assert traced_shapes == [(1, 2, 3)]
    result = infer_model_spec(weak_model, sample)
    assert result.dtype == jnp.asarray(0.0).dtype
    assert result.weak_type

    def model(x):
        return x[:, 0, :].astype(jnp.bfloat16)

    game = MaskedGame(model, FinitePlayers(6), sample, sample[0], (2, 3), BaselineMasker())
    value = jax.jit(lambda c: game(c))(game.players.full_coalition())  # noqa: PLW0108
    assert isinstance(value, VectorValue)
    assert value.dtype == game.dtype == jnp.bfloat16
    assert not game.weak_type
    converted = game.astype(jnp.float32)
    assert isinstance(converted, CastGame)
    assert converted.game.explain_index is game.explain_index
    assert converted.full_value().dtype == jnp.float32


def test_opaque_model_uses_weak_default_and_supports_a_strong_dtype_override():
    concrete_calls = []

    def model(x):
        concrete_calls.append(x.shape)
        return np.asarray(x)[:, 0]

    data = np.ones((2, 2), dtype=np.float16)
    args = (model, FinitePlayers(2), data, data[0], (2,), BaselineMasker())
    game = MaskedGame(*args)
    assert game.dtype == jnp.asarray(0.0).dtype
    assert game.weak_type and game[0].weak_type
    declared = MaskedGame(*args, output_dtype=jnp.float32)
    assert declared.dtype == jnp.float32 and not declared.weak_type
    assert concrete_calls == []
    assert declared.full_value().dtype == jnp.float32
    assert concrete_calls == [(2, 2)]
    assert game.full_value().dtype == jnp.float16

    result = infer_model_spec(np.asarray, jnp.ones((1, 2)))
    assert result.weak_type and result.dtype == jnp.asarray(0.0).dtype


def test_replace_preserves_xai_dtype_inference_and_explicit_overrides():
    data = jnp.ones((2, 2), dtype=jnp.float16)
    game = MaskedGame(lambda x: x[:, 0], FinitePlayers(2), data, data[0], (2,), BaselineMasker())
    assert game.dtype == jnp.float16
    assert replace(game).output_dtype is None

    def new_model(x):
        return x[:, 0].astype(jnp.bfloat16)

    inferred = replace(game, model=new_model)
    assert inferred.output_dtype is None
    assert inferred.dtype == inferred.full_value().dtype == jnp.bfloat16

    declared = replace(game, output_dtype="float32")
    retained = replace(declared, model=new_model)
    assert retained.dtype == retained.output_dtype == retained[0].full_value().dtype == jnp.float32


def test_masked_inference_includes_input_promotion_and_aggregation():
    data = jnp.ones((2, 2), dtype=jnp.float16)
    background = jnp.zeros((2, 2), dtype=jnp.float32)
    game = MaskedGame(lambda x: x[:, 0], FinitePlayers(2), data, background, (2,), MarginalMasker())
    assert infer_model_spec(game.model, data).dtype == jnp.float16
    assert game.dtype == game.full_value().dtype == jnp.float32
    assert not game.weak_type

    weak = replace(game, model=lambda x: jnp.full((len(x),), 2.0))
    assert infer_model_spec(weak.model, data).weak_type
    assert weak.dtype == weak.full_value().dtype
    assert not weak.weak_type
    assert not weak.full_value().weak_type


def test_structured_model_input_preserves_scalar_leaves():
    @dataclass(frozen=True, slots=True, weakref_slot=True)
    class ScaledInput(JaxAxisProtected["ScaledInput"]):
        data: jax.Array
        scale: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"data": 1, "scale": 0}
        broadcasted_batch: ClassVar[set[str]] = {"scale"}

    def predict_scaled(inputs):
        assert inputs.scale.shape == ()
        assert inputs.data.shape == (1, 2)
        return inputs.data[:, 0] * inputs.scale

    sample = ScaledInput(jnp.ones((3, 2), dtype=jnp.float16), jnp.asarray(2.0))
    assert infer_model_spec(predict_scaled, sample) == jax.ShapeDtypeStruct((), jnp.float16)


@pytest.mark.parametrize("dtype", [None, jnp.float16])
def test_masked_astype_places_only_evaluated_values_on_target_device(dtype):
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    cpu0, cpu1 = devices[:2]
    data = jax.device_put(jnp.ones((2, 2), dtype=jnp.float32), cpu0)
    game = MaskedGame(lambda x: x[:, 0], FinitePlayers(2), data, data[0], (2,), BaselineMasker())
    moved = game.astype(dtype, device=cpu1)
    assert isinstance(moved, CastGame)
    assert moved.game is game
    assert moved.game.explain_index.device == moved.game.x_explain.device == cpu0
    assert moved.game.background_index.device == moved.game.background.device == cpu0
    result = moved.full_value()
    assert result.device == cpu1
    assert moved.dtype == result.dtype == (game.dtype if dtype is None else dtype)
    assert jnp.array_equal(result, jnp.ones(2, dtype=result.dtype))


@pytest.mark.parametrize("x64", [False, True])
def test_sklearn_inference_matches_canonicalized_double_predictions(x64):
    linear_model = pytest.importorskip("sklearn.linear_model")
    data = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    model = linear_model.LinearRegression().fit(data, data[:, 0])
    with jax.enable_x64(new_val=x64):
        game = MaskedGame(model, FinitePlayers(2), data, data[0], (2,), BaselineMasker())
        want = jnp.float64 if x64 else jnp.float32
        assert game.dtype == game.full_value().dtype == want
        assert not game.weak_type


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float64"])
def test_torch_infers_dtype_without_calling_forward(dtype_name):
    torch = pytest.importorskip("torch")
    torch_dtype = getattr(torch, dtype_name)
    forward_rows = []

    class Linear(torch.nn.Linear):
        def forward(self, x):
            forward_rows.append(x.shape[0])
            return super().forward(x)

    model = Linear(2, 1, dtype=torch_dtype)
    data = torch.ones((2, 2), dtype=torch_dtype)
    game = MaskedGame(model, FinitePlayers(2), data, data[0], (2,), BaselineMasker())
    want = jax.dtypes.canonicalize_dtype(jnp.dtype(dtype_name))
    assert game.dtype == want and not game.weak_type
    assert forward_rows == []
    assert game.full_value().dtype == want
    assert forward_rows == [2]

    assert infer_model_spec(torch.nn.Identity(), data).weak_type


def test_masked_game_eager_masker_falls_back_to_weak_spec():
    class _EagerFloat32Masker(BaselineMasker):
        def aggregate(self, values):
            return jnp.asarray(np.asarray(values, dtype=np.float32)).mean(axis=-1)

    x_explain = jnp.ones((2, 2), dtype=jnp.float16)
    background = jnp.zeros((2,), dtype=jnp.float16)
    game = MaskedGame(
        lambda x: x[:, 0], FinitePlayers(2), x_explain, background, (2,), _EagerFloat32Masker()
    )
    assert game.weak_type
    assert game.full_value().dtype == jnp.float32
