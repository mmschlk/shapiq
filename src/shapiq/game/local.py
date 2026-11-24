"""Games for local explanations of machine learning models."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
from jaxtyping import Float
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game import BoolFinitePlayerGame
from shapiq.typing import typechecked


@typechecked
@dataclass(slots=True, frozen=True, weakref_slot=True)
class MLLocalXAI[P](
    BoolFinitePlayerGame[P, Float[jax.Array, "..."]], JaxAxisProtected["MLLocalXAI"]
):
    """A game reflecting the local explanations of a machine learning model."""

    model: Callable[[Float[jax.Array, "*batch n_players"]], Float[jax.Array, "*batch"]]
    x_explain: Float[jax.Array, "*batch n_players"]


# TODO(advueu963): Implemente REcontexualize and Inf TabPFN Imputer. PFN Imputer that dispatches to TabPFN and TabICL. Hooks for inf/nan imputation  # noqa: TD003
# TODO(advueu963): Local and Global (RISK) Explanation Curve of the approximators.  # noqa: TD003
# SAGE Explanations
# FIxLIP Imputer
