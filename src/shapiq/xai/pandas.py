"""pandas outputs: values from a Series or DataFrame."""

import pandas as pd
from jaxtyping import Float

from shapiq.value.value import JaxValue

from .base import as_value


@as_value.register(pd.Series | pd.DataFrame)
def _(outputs: pd.Series | pd.DataFrame) -> Float[JaxValue, " n_samples"]:
    """Return the values of a pandas object."""
    return as_value(outputs.to_numpy())
