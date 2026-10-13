"""Preprocessing shared by the tabular and TabArena loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

__all__ = ["encode_categorical", "impute"]


def impute(x: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric columns with their median and other columns with their mode."""
    for column in x.columns:
        if x[column].isna().any():
            if pd.api.types.is_numeric_dtype(x[column]):
                x[column] = x[column].fillna(x[column].median())
            else:
                x[column] = x[column].fillna(x[column].mode()[0])
    return x


def encode_categorical(x: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode all text and category columns (missing and unknown categories become ``-1``)."""
    categorical = x.select_dtypes(include=["object", "category", "string"]).columns
    if len(categorical) == 0:
        return x
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
    )
    x = x.copy()
    values = x[categorical].astype(object)
    x[categorical] = encoder.fit_transform(values.where(values.notna(), np.nan))  # pd.NA -> NaN
    return x
