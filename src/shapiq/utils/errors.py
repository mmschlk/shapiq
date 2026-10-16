"""This module contains error and warnings for the shapiq package."""

from __future__ import annotations


def raise_deprecation_warning(
    message: str,
    deprecated_in: str,
    removed_in: str,
) -> None:
    """Raise a deprecation warning with the given details.

    Args:
        message: What is deprecated and what to use instead.
        deprecated_in: The version in which the feature was deprecated.
        removed_in: The version in which the feature will be removed.
    """
    from warnings import warn

    message += (
        f" This feature is deprecated in version {deprecated_in} and will be removed in version "
        f"{removed_in}."
    )

    warn(message, DeprecationWarning, stacklevel=2)
