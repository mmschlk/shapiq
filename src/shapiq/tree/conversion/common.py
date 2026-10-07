"""Top-level dispatch for converting any supported tree-based model to the unified internal format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import LazyType, lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.tree.base import TreeModel


@lazydispatch
def conversion_generator(
    model: object, class_label: int | None = None
) -> TreeModel | list[TreeModel]:
    """Dispatch conversion of a tree-based model to its registered handler.

    Raises ``NotImplementedError`` for any model type that has not been registered via
    :func:`register`.  Concrete handlers are registered in the ``sklearn`` and ``boosting``
    sub-modules.

    Args:
        model: The tree-based model to convert.
        class_label: The class label to explain. Only relevant for multi-class classification
            models.

    Returns:
        A single ``TreeModel`` for single-tree models, or a ``list[TreeModel]`` for ensemble
        models (one entry per tree).

    Raises:
        NotImplementedError: If no conversion handler has been registered for ``type(model)``.
    """
    msg = f"Conversion for model type {type(model)} is not implemented."
    raise NotImplementedError(msg)


def register(cls: LazyType, func: Callable) -> None:
    """Register a conversion function for a given model type.

    Associates ``func`` with ``cls`` in the ``conversion_generator`` dispatch table so that
    :func:`convert_tree_model` will call ``func`` when passed an instance of ``cls``.

    Args:
        cls: The model class (or lazy type string) to register a handler for.
        func: The conversion callable that accepts a model instance and returns a
            ``TreeModel`` or ``list[TreeModel]``.
    """
    conversion_generator.register(cls=cls, func=func)


def select_binary_class_margin(trees: list[TreeModel], class_label: int | None) -> list[TreeModel]:
    """Return the raw margin of ``class_label`` for a binary classifier with a single raw output.

    Binary gradient boosting classifiers model one raw output, the log-odds of class ``1``. The
    log-odds of class ``0`` are its negation, so for ``class_label=0`` the leaf values (which
    already include any base score or initial-prediction offset) and the empty predictions are
    negated in place. ``class_label=None`` selects class ``1``.

    Args:
        trees: The converted trees modeling the class-``1`` raw output.
        class_label: The class to explain: ``0``, ``1``, or ``None`` (class ``1``).

    Returns:
        The trees modeling the raw output of the selected class.

    Raises:
        ValueError: If ``class_label`` is not ``None``, ``0``, or ``1``.
    """
    if class_label is None or class_label == 1:
        return trees
    if class_label != 0:
        msg = f"class_label={class_label} is invalid for a binary classifier; use 0 or 1."
        raise ValueError(msg)
    for tree in trees:
        tree.values = -tree.values
        tree.empty_prediction = -tree.empty_prediction
    return trees


def convert_tree_model(model: object, class_label: int | None = None) -> list[TreeModel]:
    """Convert a tree-based model to the unified internal tree format used by shapiq.

    Delegates to the appropriate registered conversion handler via
    :func:`conversion_generator`.

    Args:
        model: The tree-based model to convert. Supported types include scikit-learn decision
            trees and random forests, XGBoost models, LightGBM models, and CatBoost models.
        class_label: The class label to explain for multiclass gradient-boosted models.
            ``None`` is passed to backend converters as their default class selection.

    Returns:
        A list of ``TreeModel`` objects. Single-tree converters are normalized to a one-item
        list by their handlers; ensemble converters return one entry per selected tree.

    Raises:
        NotImplementedError: If no conversion handler has been registered for ``type(model)``.
    """
    return conversion_generator(model, class_label=class_label)
