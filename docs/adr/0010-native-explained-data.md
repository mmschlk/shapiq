# Native Explained Data

**Status: Accepted.** Refines the backend boundary of ADR 0002 for local explanation games.

## Decision

Local explanation games keep `x_explain` and `background` in the format the user
passes, e.g. a pandas DataFrame, a NumPy array or a torch tensor, stored unchanged
as static dataclass metadata. Like torch batches, the data has the shape
`(*batch, *input_shape)`, with `input_shape` given by the user: tables `(n_features,)`,
images `(channels, height, width)`. A masker's background may add sample axes before
the input axes, e.g. one for averaged background rows.

The game batch comes from the data. Every dataset gets a JAX index of its batch
entries, built at construction and marked as a broadcast field, so the batches of
the explained inputs and of the backgrounds broadcast into the game batch.
Indexing, broadcasting and stacking games act on these indices, never on the data.
Only for the model call are the batch and sample axes merged into one; the input
axes are never touched.

Maskers turn coalitions into model inputs in the data's own format, dispatching on
the data type and staying on the data's device, so the model always receives inputs
it supports. `predict` calls the model on them and `as_value` converts the outputs
into game values.

Tracing is decided by the backends, not the games. Over JAX data and a JAX model,
masking and the model call trace, so such a game can be evaluated under `jit`. Other
data and models, e.g. NumPy, pandas, scikit-learn or torch, are called eagerly: their
`impute`, `select` and `predict` registrations raise a clear error under tracing,
instead of failing on a tracer deep inside the library. Games that are eager by nature,
e.g. recontextualization refitting the model per coalition, raise likewise. How to call
a model efficiently, e.g. in large batches or on an accelerator, is model-specific and
belongs to its `predict` registration.

## Trade-Off

Categorical columns, dtypes, devices and feature names reach the model unchanged,
and every model is called the way it is fastest. In exchange, games over non-JAX data
or models cannot be traced, nor differentiated through the model. Games cannot be
passed to `jit` as arguments, even over JAX data, since the data is static metadata
and arrays are unhashable; `jit` takes them in a closure instead.
