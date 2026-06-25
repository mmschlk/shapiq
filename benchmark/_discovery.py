"""Shared SV approximator discovery + SV-mode construction.

Single source of truth for the cross-method test harness
(``tests/shapiq/tests_unit/tests_approximators/test_approximators_vs_exact.py``)
and the benchmark CLI (``benchmark.performance``). Both consume the same
helpers so that the test contract and the benchmark run stay aligned.
"""

from __future__ import annotations

import importlib
from typing import Any


# Project-specific additions that may not yet be registered in
# ``SV_APPROXIMATORS`` on every feature branch. Surfacing them by name lets
# the runner report ``skipped:not_registered`` on branches that do not ship
# the class yet, instead of silently omitting it.
PROJECT_APPROXIMATOR_NAMES: tuple[str, ...] = (
    "LeverageSHAP",
    "PolySHAPKAdd",
    "PolySHAPPartial",
    "OddSHAP",
)


# Approximators that are registered in ``SV_APPROXIMATORS`` but deliberately
# kept out of the cross-method benchmark/conformance suite. ``PolySHAPPrior``
# needs a handcrafted ``q_prior`` (per-game prior interaction knowledge), so a
# generic budget-vs-error sweep does not give it a meaningful, comparable input.
EXCLUDED_APPROXIMATOR_NAMES: frozenset[str] = frozenset({
    "PolySHAPPrior",
})


# Method-specific construction overrides for classes whose constructor needs
# more than ``(n=n, random_state=...)``. Each entry is callable
# ``(n: int) -> dict[str, Any]`` returning the extra kwargs to merge in.
_SV_CONSTRUCT_OVERRIDES: dict[str, Any] = {
    "PolySHAPKAdd": lambda _n: {"max_order": 1},
    "PolySHAPPartial": lambda n: {"n_explanation_terms": n + 1},
}


def discover_sv_approximator_names() -> list[str]:
    """Return SV approximator class names to test/benchmark.

    Sources:
    - ``shapiq.approximator.SV_APPROXIMATORS`` (the canonical registry).
    - Project-specific additions from ``PROJECT_APPROXIMATOR_NAMES``.

    Names in ``EXCLUDED_APPROXIMATOR_NAMES`` are dropped. Duplicates removed
    while preserving registry order.
    """
    module = importlib.import_module("shapiq.approximator")
    registry = getattr(module, "SV_APPROXIMATORS", [])
    existing = [cls.__name__ for cls in registry]
    names = dict.fromkeys(existing + list(PROJECT_APPROXIMATOR_NAMES))
    return [n for n in names if n not in EXCLUDED_APPROXIMATOR_NAMES]


def load_approximator(name: str):
    """Return the named class from ``shapiq.approximator`` or ``None``."""
    module = importlib.import_module("shapiq.approximator")
    return getattr(module, name, None)


def construct_for_sv(
    approx_cls,
    n: int,
    random_state: int,
) -> tuple[Any, Exception | None]:
    """Build an SV-mode estimator.

    Returns ``(estimator, None)`` on success or ``(None, exc)`` on failure.

    Construction strategy, in order:

    1. A method-specific override from ``_SV_CONSTRUCT_OVERRIDES`` (covers
       classes like ``PolySHAPKAdd`` whose constructor requires
       ``max_order`` or ``n_explanation_terms``).
    2. The multi-index signature
       ``Approx(n=n, index='SV', max_order=1, random_state=...)`` —
       covers ``SPEX / ProxySPEX / ProxySHAP / RegressionMSR / kADDSHAP``
       which default to ``index='FBII'`` with ``max_order=n``.
    3. The minimal SV-only signature ``Approx(n=n, random_state=...)`` —
       covers ``KernelSHAP / OwenSamplingSV / SVARM / etc.``

    A ``ValueError`` from inside a constructor that *accepted* our kwargs
    is more informative than a ``TypeError`` from a signature mismatch —
    it means "the signature matched, the implementation is broken". The
    first ``ValueError`` wins over any subsequent ``TypeError`` when
    reporting.
    """
    name = approx_cls.__name__
    override = _SV_CONSTRUCT_OVERRIDES.get(name)
    candidate_kwargs: list[dict[str, Any]] = []
    if override is not None:
        extra = override(n) if callable(override) else override
        candidate_kwargs.append({"n": n, "random_state": random_state, **extra})
    candidate_kwargs.extend([
        dict(n=n, index="SV", max_order=1, random_state=random_state),
        dict(n=n, random_state=random_state),
    ])

    first_value_error: Exception | None = None
    last_exc: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            return approx_cls(**kwargs), None
        except TypeError as exc:
            last_exc = exc
        except (ValueError, ImportError) as exc:
            if first_value_error is None:
                first_value_error = exc
            last_exc = exc
    return None, first_value_error if first_value_error is not None else last_exc


def safe_approximate(
    estimator,
    budget: int,
    game,
) -> tuple[Any, Exception | None]:
    """Call ``estimator.approximate(budget, game)``.

    Returns ``(iv, None)`` on success or ``(None, exc)`` when the
    estimator explicitly refuses the regime (``ValueError`` or
    ``RuntimeError``).
    """
    try:
        return estimator.approximate(budget, game), None
    except (ValueError, RuntimeError) as exc:
        return None, exc
