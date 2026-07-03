"""Cross-method performance benchmark for SV approximators.

Public CLI lives in :mod:`benchmark.performance`. Shared discovery and
SV-mode construction utilities — consumed by both the CLI and the
pytest conformance harness — live in :mod:`benchmark._discovery`.

Invoke as a module to keep relative imports working::

    python -m benchmark.performance --check
"""
