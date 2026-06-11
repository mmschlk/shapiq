"""Hamming-kernel GP surrogate over binary coalition vectors.

Flat, self-contained surrogate for the ShaplEIG approximator (the research
repository uses a richer surrogate hierarchy; this copy keeps only what
ShaplEIG needs). It fits directly on binary coalition vectors — no input
transform — and exposes the fitted state as plain tensors via
:meth:`HammingGP.tensors`, which is the only surface the Shapley math core
consumes.

Maintainer note: this module is a controlled copy of the validated reference
implementation from the ShaplEIG research codebase (the code accompanying the
paper). Changes are developed and validated there first, then ported here.
"""

from __future__ import annotations

import logging
import math

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms import Standardize
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from ._shapley_math import GPTensors

log = logging.getLogger(__name__)


class HammingGP:
    """Exact single-output GP with a scaled Hamming product kernel."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        *,
        fixed_noise_level: float | None = 1e-6,
        min_lengthscale: float = 1e-6,
        min_inferred_noise_level: float = 1e-4,
        amount_restarts: int = 5,
    ) -> None:
        """Build the GP on binary coalitions ``train_X`` and values ``train_Y``.

        Args:
            train_X: Binary coalition matrix of shape ``(t, n_players)``.
            train_Y: Coalition values of shape ``(t, 1)``.
            fixed_noise_level: Fixed observation noise for (near-)noiseless
                games, in STANDARDIZED output units (i.e. relative to Var(y)):
                the GP runs with exactly this noise variance after output
                standardization, independent of the output scale. Must be
                >= 1e-6 (gpytorch's hard floor). ``None`` infers the noise via
                a ``GaussianLikelihood``.
            min_lengthscale: Lower bound for the per-player ARD lengthscales.
            min_inferred_noise_level: Lower noise bound when noise is learned.
            amount_restarts: Maximum number of L-BFGS-B fitting attempts.
        """
        self._init_kwargs: dict = {
            "fixed_noise_level": fixed_noise_level,
            "min_lengthscale": min_lengthscale,
            "min_inferred_noise_level": min_inferred_noise_level,
            "amount_restarts": amount_restarts,
        }
        self.amount_restarts = amount_restarts

        d = train_X.shape[-1]
        lengthscale_prior = LogNormalPrior(loc=math.sqrt(2) + math.log(d) * 0.5, scale=math.sqrt(3))
        # Over binary inputs, botorch's CategoricalKernel is exactly the
        # weighted Hamming product kernel k(x,x') = exp(-mean_d 1[x_d != x'_d] / l_d).
        base_kernel = CategoricalKernel(
            ard_num_dims=d,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                min_lengthscale,
                initial_value=max(lengthscale_prior.mode, min_lengthscale),
            ),
        )
        covar_module = ScaleKernel(base_kernel)

        if fixed_noise_level is not None:
            # Pre-scale by std(y)^2 (replicating botorch's Standardize stdvs,
            # incl. its small-std fallback) so that the noise is EXACTLY
            # `fixed_noise_level` in standardized units after the transform.
            stdvs = train_Y.std(dim=0, keepdim=True)
            stdvs = torch.where(stdvs >= 1e-8, stdvs, torch.ones_like(stdvs))
            train_Yvar = torch.full_like(train_Y, fixed_noise_level) * stdvs.pow(2)
            likelihood = None
        else:
            train_Yvar = None
            noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    min_inferred_noise_level,
                    initial_value=max(noise_prior.mode, min_inferred_noise_level),
                ),
            )

        self._model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )

    def fit(self) -> None:
        """MAP hyperparameter fit (L-BFGS-B with restarts)."""
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        try:
            # Stop at the first successful attempt (do not run all attempts
            # and pick the best) — as in the reference implementation.
            fit_gpytorch_mll(
                mll,
                max_attempts=self.amount_restarts,
                pick_best_of_all_attempts=False,
            )
        except Exception:
            # Keep running with the current hyperparameters, but make the
            # failure visible.
            log.exception(
                "GP hyperparameter fit failed (with %d attempts); continuing "
                "with current hyperparameters.",
                self.amount_restarts,
            )

    def update_data(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Rebuild on new data, warmstarting from the current hyperparameters."""
        old_model = self._model
        was_eval = not old_model.training
        rebuilt = HammingGP(train_X, train_Y, **self._init_kwargs)
        source = dict(old_model.named_parameters())
        with torch.no_grad():
            for name, target in rebuilt._model.named_parameters():
                src = source.get(name)
                if src is not None and src.shape == target.shape:
                    target.copy_(src.detach())
        if was_eval:
            rebuilt._model.eval()
        self.__dict__.update(rebuilt.__dict__)

    def posterior_variance_diag(
        self, X: torch.Tensor, *, observation_noise: bool = False
    ) -> torch.Tensor:
        """Marginal posterior variances of ``X`` via the gpytorch posterior."""
        with torch.no_grad():
            posterior = self._model.posterior(X, observation_noise=observation_noise)
            mvn = posterior.mvn  # ty: ignore[unresolved-attribute]
            return mvn.lazy_covariance_matrix.diagonal(dim1=-2, dim2=-1)

    def tensors(self) -> GPTensors:
        """Plain-tensor view of the fitted surrogate, consumed by the math core."""
        return GPTensors.from_botorch(self._model)
