"""Numerical precision guards for the polynomial-based tree explainers.

The path-dependent polynomial algorithms (:class:`~shapiq.tree.treeshapiq.TreeSHAPIQ` and
:class:`~shapiq.tree.linear.computer.LinearTreeSHAP`) lose precision as the number of
*distinct features along a single root-to-leaf path* grows: their float64 round-off grows by
roughly one order of magnitude for every two additional features per path, independent of how
the interpolation systems are solved (see https://github.com/mmschlk/shapiq/issues/545). Tree
depth itself is not the limit; a depth-100 tree whose paths each use at most 20 distinct
features is computed accurately.

This module centralises the guard applied at explainer construction time: a
:class:`TreeNumericalPrecisionWarning` inside the degrading band and a
:class:`TreeNumericalPrecisionError` where values would be meaningless. The
:class:`~shapiq.tree.quadrature.QuadratureTreeSHAP` explainer is unaffected by this limit.
"""

from __future__ import annotations

WARN_FEATURES_PER_PATH = 24
"""From this many distinct features per decision path, values degrade measurably."""

ERROR_FEATURES_PER_PATH = 30
"""From this many distinct features per decision path, values are unreliable and refused."""


class TreeNumericalPrecisionWarning(RuntimeWarning):
    """The polynomial tree explainers are operating in their degrading precision band."""


class TreeNumericalPrecisionError(ValueError):
    """The polynomial tree explainers cannot compute reliable values for this tree."""


def check_features_per_path(max_features_per_path: int, *, algorithm: str) -> None:
    """Guard the polynomial explainers against numerically unreliable trees.

    Args:
        max_features_per_path: The maximum number of distinct features along any root-to-leaf
            path of the tree (``max(edge_heights)`` of the edge tree).
        algorithm: Name of the calling algorithm, used in the message.

    Raises:
        TreeNumericalPrecisionError: If the tree exceeds :data:`ERROR_FEATURES_PER_PATH`.

    Warns:
        TreeNumericalPrecisionWarning: If the tree reaches :data:`WARN_FEATURES_PER_PATH`.

    """
    if max_features_per_path >= ERROR_FEATURES_PER_PATH:
        msg = (
            f"This tree has a decision path using {max_features_per_path} distinct features; "
            f"{algorithm} values are numerically unreliable beyond "
            f"{ERROR_FEATURES_PER_PATH - 1} (https://github.com/mmschlk/shapiq/issues/545). "
            "Tree depth itself is not the limit, only distinct features per path. "
            "Use QuadratureTreeSHAP (the TreeExplainer default) or constrain the tree."
        )
        raise TreeNumericalPrecisionError(msg)
    if max_features_per_path >= WARN_FEATURES_PER_PATH:
        import warnings

        expected_error = (max_features_per_path - 34) / 2
        msg = (
            f"A decision path in this tree uses {max_features_per_path} distinct features; "
            f"{algorithm} values are expected to carry a relative error of roughly "
            f"1e{expected_error:.0f}. From {ERROR_FEATURES_PER_PATH} features per path the "
            "computation is refused; QuadratureTreeSHAP (the TreeExplainer default) computes "
            "exact values."
        )
        warnings.warn(msg, TreeNumericalPrecisionWarning, stacklevel=3)
