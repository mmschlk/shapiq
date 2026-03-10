"""Lazy alternatives to singledispatch and isinstance."""

from lazy_dispatch.isinstance import LazyType, lazy_isinstance, lazy_issubclass
from lazy_dispatch.load import lazy_callable, lazy_import
from lazy_dispatch.singledispatch import is_valid_dispatch_type, lazydispatch

__all__ = [
    "LazyType",
    "is_valid_dispatch_type",
    "lazy_callable",
    "lazy_import",
    "lazy_isinstance",
    "lazy_issubclass",
    "lazydispatch",
]
