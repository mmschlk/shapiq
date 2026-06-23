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
# the class yet, instead of silently omitting it. ``PolySHAP`` is split on
# its feature branch into three variants (KAdd / Partial / Prior); the
# umbrella name is kept so each appears in ``--check`` independently.
PROJECT_APPROXIMATOR_NAMES: tuple[str, ...] = (
    "LeverageSHAP",
    "OptimizedKernelSHAP",
    "PolySHAP",
    "PolySHAPKAdd",
    "PolySHAPPartial",
    "PolySHAPPrior",
    "OddSHAP",
)


# Method-specific construction overrides for classes whose constructor needs
# more than ``(n=n, random_state=...)``. Each entry is callable
# ``(n: int) -> dict[str, Any]`` returning the extra kwargs to merge in.
_SV_CONSTRUCT_OVERRIDES: dict[str, Any] = {
    "PolySHAPKAdd": lambda _n: {"max_order": 1},
    "PolySHAPPartial": lambda n: {"n_explanation_terms": n + 1},
    "PolySHAPPrior": lambda n: {"q_prior": [(i,) for i in range(n)]},
}


def discover_sv_approximator_names() -> list[str]:
    """Return SV approximator class names to test/benchmark.

    Sources:
    - ``shapiq.approximator.SV_APPROXIMATORS`` (the canonical registry).
    - Project-specific additions from ``PROJECT_APPROXIMATOR_NAMES``.

    Duplicates removed while preserving registry order.
    """
    module = importlib.import_module("shapiq.approximator")
    registry = getattr(module, "SV_APPROXIMATORS", [])
    existing = [cls.__name__ for cls in registry]
    return list(dict.fromkeys(existing + list(PROJECT_APPROXIMATOR_NAMES)))


def load_approximator(name: str):
    """Return the named class from ``shapiq.approximator`` or ``None``."""
    module = importlib.import_module("shapiq.approximator")

    if name == "OptimizedKernelSHAP":
        BaseKernelSHAP = getattr(module, "KernelSHAP", None)
        if BaseKernelSHAP is None:
            return None

        class OptimizedKernelSHAP(BaseKernelSHAP):
            def __init__(self, *args, **kwargs):
                kwargs["pairing_trick"] = True
                super().__init__(*args, **kwargs)

        return OptimizedKernelSHAP

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
       ``max_order``, ``n_explanation_terms``, or ``q_prior``).
    2. The multi-index signature
       ``Approx(n=n, index='SV', max_order=1, random_state=...)`` —
       covers ``SPEX / ProxySPEX / ProxySHAP / MSRBiased / kADDSHAP``
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
    candidate_kwargs.extend(
        [
            dict(n=n, index="SV", max_order=1, random_state=random_state),
            dict(n=n, index="SV", random_state=random_state),
            dict(n=n, random_state=random_state),
        ]
    )

    first_value_error: Exception | None = None
    last_exc: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            return approx_cls(**kwargs), None
        except TypeError as exc:
            last_exc = exc
        except ValueError as exc:
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
