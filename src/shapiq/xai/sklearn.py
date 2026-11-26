"""Scikit-learn prediction and output dtype metadata."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Float
from sklearn.base import BaseEstimator

from shapiq.masker.base import require_eager
from shapiq.value.value import JaxValue

from .base import as_value, infer_model_spec, predict


@predict.register(BaseEstimator)
def _(model: Any, x: Any) -> Float[JaxValue, " n_samples"]:  # noqa: ANN401
    """Return class probabilities, decision scores, or predictions."""
    require_eager("scikit-learn models", x)
    if hasattr(model, "predict_proba"):
        return as_value(model.predict_proba(x))
    if hasattr(model, "decision_function"):
        return as_value(model.decision_function(x))
    return as_value(model.predict(x))


@infer_model_spec.register(BaseEstimator)
def _(model: BaseEstimator, x_sample: object = None) -> jax.ShapeDtypeStruct:  # noqa: ARG001
    """Estimate output dtype using scikit-learn's usual double precision."""
    return jax.ShapeDtypeStruct((), jax.dtypes.canonicalize_dtype(jnp.dtype(jnp.float64)))
