"""ShaplEIG: Bayesian experimental design approximator for Shapley values.

Requires the optional ``shapleig`` extra (``pip install shapiq[shapleig]``).
"""

from .shapleig import ShaplEIG

__all__ = ["ShaplEIG"]
