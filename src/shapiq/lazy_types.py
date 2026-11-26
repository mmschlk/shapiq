"""Canonical backend types for dispatch without optional imports."""

TORCH_MODULE = "torch.nn.modules.module.Module"
TORCH_TENSOR = "torch.Tensor"
FLAX_MODULE = "flax.nnx.module.Module"
SKLEARN_MODULE = "sklearn.base.BaseEstimator"
PANDAS_DATAFRAME = "pandas.core.frame.DataFrame"
PANDAS_SERIES = "pandas.core.series.Series"
PANDAS_OUTPUT = (PANDAS_SERIES, PANDAS_DATAFRAME)

JAX_ARRAY = "jax.Array"
JAX_TRACER = "jax._src.core.Tracer"
JAX_ARRAY_LIKE = "ndlike.jax_like.JaxLikeImplementation"
JAX_VALUE = (JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER)
