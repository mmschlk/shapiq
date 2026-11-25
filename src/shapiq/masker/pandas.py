"""Masking of pandas DataFrames, column by column.

A DataFrame is a batch of rows with one input axis, its columns. A background with a sample axis
is a single entry: its rows are the samples.
"""

from typing import Any

import jax
import numpy as np
import pandas as pd
from jaxtyping import Bool, Int

from .base import impute, require_eager, select


@impute.register(pd.DataFrame)
def impute_dataframe(
    x_explain: pd.DataFrame,
    background: pd.DataFrame,
    explain_rows: Int[jax.Array, " n"],
    background_rows: Int[jax.Array, " n"],
    keep: Bool[jax.Array, "n n_features"],
    *,
    sample_ndim: int,
) -> pd.DataFrame:
    """Impute every column on its own, keeping its dtype, e.g. categories."""
    require_eager("DataFrame data", explain_rows, background_rows, keep)
    rows, mask = np.asarray(explain_rows), np.asarray(keep)
    columns: dict[Any, pd.Series] = {}
    for j, name in enumerate(x_explain.columns):
        picked = x_explain[name].to_numpy()[rows]
        if sample_ndim:  # every explained row against every background row
            filled = np.where(mask[:, None, j], picked[:, None], background[name].to_numpy()[None])
        else:  # every explained row against its baseline row
            reference = background[name].to_numpy()[np.asarray(background_rows)]
            filled = np.where(mask[:, j], picked, reference)
        columns[name] = pd.Series(filled.reshape(-1), dtype=x_explain[name].dtype)
    return pd.DataFrame(columns)


@select.register(pd.DataFrame)
def select_dataframe(
    data: pd.DataFrame, rows: Int[jax.Array, " n"], features: Bool[jax.Array, " n_features"]
) -> pd.DataFrame:
    """Select by position, keeping every column's dtype and name."""
    require_eager("DataFrame data", rows, features)
    return data.iloc[np.asarray(rows), np.flatnonzero(np.asarray(features))]
