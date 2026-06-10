"""This module contains error and warnings for the shapiq package."""

from __future__ import annotations


def raise_deprecation_warning(
    message: str,
    deprecated_in: str,
    removed_in: str,
) -> None:
    """Raise a deprecation warning with the given details."""
    from warnings import warn

    message += (
        f" This feature is deprecated in version {deprecated_in} and will be removed in version "
        f"{removed_in}."
    )

    warn(message, DeprecationWarning, stacklevel=2)


class RepresentationLimitError(ValueError):
    """The interpolation degree exceeds what the float64 TreeSHAP pipeline can evaluate.

    The exact interpolation coefficients are available at any depth, but the
    downstream double-precision evaluation would cancel them beyond an acceptable
    error. For order-1 SV/SII configurations, ``TreeExplainer`` catches this and
    re-routes the affected tree to ``TreeSHAPIQ`` (whose interpolation degree is
    bounded by the features in the tree rather than the tree depth); in every
    other configuration — higher orders, the feature-bounded indices, or trees
    that exceed the limit on both paths — the error propagates from the
    constructor.
    """
