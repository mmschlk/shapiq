r"""Shapley-value math core for ShaplEIG (ESP-accelerated, tensor-in/tensor-out).

Maintainer note: this module is a controlled copy of the validated reference
implementation from the ShaplEIG research codebase (the code accompanying the
paper). Changes are developed and validated there first, then ported here
verbatim.

All functions are pure: they take plain tensors (kernel lengthscales, outputscale,
binary training coalitions, standardized targets, noise) and never touch a GP
object. The Shapley weights are baked into the generating-polynomial (elementary
symmetric polynomial, ESP) math — no explicit affine matrix ``A`` or coalition
enumeration ``Z`` is ever formed, so everything is O(poly(p)) in the number of
players regardless of the 2^p coalition space.

Notation (paper: "ShaplEIG: Bayesian Experimental Design for Shapley Value
Estimation"):

- ``p``: number of players; ``t``: number of queried coalitions (training points).
- :math:`A`: the (:math:`p \times 2^p`) Shapley affine operator (implicit).
- :math:`K`: weighted Hamming product kernel over binary coalitions.
- ``A_KZW`` :math:`= A K(Z, W)` (:math:`p \times |W|`), computed via ESP.
- ``AKZZA`` :math:`= A K(Z, Z) A^\top` (:math:`p \times p`), computed via ESP.
- ``AEA`` :math:`= A \Sigma_\nu A^\top` (unscaled posterior property covariance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    from linear_operator.utils.cholesky import psd_safe_cholesky
except ImportError as err:
    from ._error import _shapleig_import_error

    raise _shapleig_import_error from err

DTYPE = torch.float64

__all__ = [
    "GPTensors",
    "ShaplEIGCache",
    "a_kzw",
    "a_sigma_w",
    "aea",
    "affine_posterior_mean",
    "akzza",
    "hamming_kernel",
    "kernel_alpha_beta",
    "noisy_posterior_variance",
    "shapleig_utilities",
]


# ---------------------------------------------------------------------------
# Plain tensors extracted from a fitted GP (the §4.0 seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPTensors:
    """Everything the SV math needs from a fitted Hamming GP, as plain tensors.

    All quantities live in the *standardized* output space of the GP except
    ``emp_mean`` / ``emp_std`` which describe the standardization itself.
    """

    train_X: torch.Tensor  # (t, p) binary coalitions
    train_y_std: torch.Tensor  # (t,) standardized targets
    lengthscales: torch.Tensor  # (p,)
    outputscale: torch.Tensor  # scalar
    noise: torch.Tensor  # scalar likelihood noise (standardized space)
    mean_const: torch.Tensor  # scalar constant mean (standardized space)
    emp_mean: torch.Tensor  # scalar empirical mean of raw targets
    emp_std: torch.Tensor  # scalar empirical std of raw targets

    @staticmethod
    def from_botorch(model) -> GPTensors:
        """Extract tensors from a (botorch) ``SingleTaskGP``-like model.

        This is the only place that reaches into model internals; the math
        below never does.
        """
        with torch.no_grad():
            noise = model.likelihood.noise.detach().reshape(-1).to(DTYPE)
            assert torch.allclose(noise, noise[0]), (
                "GPTensors assumes homoskedastic noise; got per-point noise levels that differ."
            )
            train_X = model.train_inputs[0].detach().to(DTYPE)
            if hasattr(model, "input_transform"):
                # The stored train_inputs are already transformed in eval mode;
                # in train mode botorch stores raw inputs. Normalize via the
                # transform to be safe (idempotent for binary identity cases).
                train_X = model.input_transform(train_X).to(DTYPE)
            return GPTensors(
                train_X=train_X,
                train_y_std=model.train_targets.detach().to(DTYPE).reshape(-1),
                lengthscales=model.covar_module.base_kernel.lengthscale.detach()
                .reshape(-1)
                .to(DTYPE),
                outputscale=model.covar_module.outputscale.detach().to(DTYPE),
                noise=noise[0],
                mean_const=model.mean_module.constant.detach().to(DTYPE),
                emp_mean=model.outcome_transform.means.detach().reshape(-1)[0].to(DTYPE),
                emp_std=model.outcome_transform.stdvs.detach().reshape(-1)[0].to(DTYPE),
            )

    @property
    def p(self) -> int:
        """Number of players (input dimension)."""
        return self.train_X.shape[1]


# ---------------------------------------------------------------------------
# Kernel primitives
# ---------------------------------------------------------------------------


def kernel_alpha_beta(
    lengthscales: torch.Tensor, outputscale: torch.Tensor, dtype: torch.dtype = DTYPE
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Per-player match/mismatch kernel factors of the Hamming product kernel.

    The weighted Hamming kernel
    :math:`k(z, w) = s \exp(-\mathrm{mean}_d\, \mathbb{1}[z_d \neq w_d]/\ell_d)`
    factorizes as :math:`\prod_d f_d` with :math:`f_d = \alpha_d` if
    :math:`z_d = w_d` else :math:`\beta_d`, where the outputscale :math:`s` is
    distributed evenly across players (:math:`\alpha_d = s^{1/p}`).
    """
    ls = lengthscales.detach().reshape(-1).to(dtype)
    p = ls.numel()
    s = outputscale.detach().to(dtype=dtype, device=ls.device)
    factor = s.pow(1.0 / p)
    alpha = torch.full((p,), factor.item(), dtype=dtype, device=ls.device)
    beta = torch.exp(-torch.ones(p, dtype=dtype, device=ls.device) / (ls * p)) * factor
    return alpha, beta


def hamming_kernel(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscales: torch.Tensor,
    outputscale: torch.Tensor,
) -> torch.Tensor:
    r"""Dense weighted Hamming product kernel matrix :math:`s \exp(-\mathrm{mean}(\delta/\ell))`.

    Pure-tensor counterpart of the surrogate's kernel module (botorch's
    ``CategoricalKernel``): the math core takes plain lengthscale/outputscale
    tensors and never touches gpytorch kernel objects.
    """
    ls = lengthscales.reshape(-1).to(X1.dtype)
    delta = (X1.unsqueeze(-2) != X2.unsqueeze(-3)).to(X1.dtype)
    dists = (delta / ls).mean(-1)
    return outputscale.to(X1.dtype) * torch.exp(-dists)


# ---------------------------------------------------------------------------
# ESP-accelerated A · K(Z, W)
# ---------------------------------------------------------------------------


@torch.no_grad()
def a_kzw(
    W: torch.Tensor,
    lengthscales: torch.Tensor,
    outputscale: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    r"""Compute :math:`A K(Z, W)` for binary coalitions ``W`` (shape (T, p)), giving (p, T).

    **Vanilla (paper) ESP route, in coefficient representation**: for each
    :math:`w`, the generating polynomial :math:`\prod_r (\gamma_r + \delta_r \zeta)`
    is built incrementally as prefix and suffix coefficient tables
    (max-normalized per step, log-scale tracked); per player :math:`j` the
    :math:`\mathrm{prefix}(j)\,\mathrm{suffix}(j+1)` product is formed by
    convolution and contracted with the Shapley size weights (Theorem B.1).
    This is the variant published with the paper.
    """
    assert W.ndim == 2
    alpha, beta = kernel_alpha_beta(lengthscales, outputscale)
    chunks = []
    for start in range(0, W.shape[0], chunk_size):
        chunks.append(_a_kz_batch_coeff(W[start : start + chunk_size], alpha, beta))
    return torch.cat(chunks, dim=1)


def _a_kz_batch_coeff(batch: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    dtype = alpha.dtype
    device = alpha.device
    batch = batch.to(device=device, dtype=dtype)
    batch_size, p = batch.shape
    tiny = torch.finfo(dtype).tiny

    gamma = torch.where(batch == 0, alpha.unsqueeze(0), beta.unsqueeze(0))
    delta = torch.where(batch == 0, beta.unsqueeze(0), alpha.unsqueeze(0))
    w_in, w_out = _shapley_weights_signed(p, dtype=dtype, device=device)

    # Step 1: prefix and suffix coefficient tables across factor products
    # (forward and backward direction), max-normalized with log-scale tracking.
    def get_iter_coeff_tables(suffix: bool = False):
        _coeff_tables: list[torch.Tensor] = [torch.empty(0)] * (p + 1)
        _log_scales = torch.zeros((p + 1, batch_size), dtype=dtype, device=device)
        _coeff_tables[0 if not suffix else p] = torch.ones(
            (batch_size, 1), dtype=dtype, device=device
        )
        _range = range(p) if not suffix else range(p - 1, -1, -1)
        for r in _range:
            prev_index = r if not suffix else r + 1
            next_index = r + 1 if not suffix else r
            prev_table = _coeff_tables[prev_index]
            curr_table = torch.zeros(
                (batch_size, prev_table.shape[1] + 1), dtype=dtype, device=device
            )
            curr_table[:, :-1] += gamma[:, r : r + 1] * prev_table
            curr_table[:, 1:] += delta[:, r : r + 1] * prev_table
            scales = curr_table.abs().amax(dim=1).clamp_min(tiny)
            _coeff_tables[next_index] = curr_table / scales.unsqueeze(1)
            _log_scales[next_index, :] = _log_scales[prev_index, :] + torch.log(scales)
        return _coeff_tables, _log_scales

    prefix_coeffs, prefix_log_scales = get_iter_coeff_tables(suffix=False)
    suffix_coeffs, suffix_log_scales = get_iter_coeff_tables(suffix=True)

    # Step 2: per player j, convolve prefix(j) with suffix(j+1) and contract
    # with the Shapley weights (Theorem B.1), undoing the normalization scales.
    out = torch.empty((p, batch_size), dtype=dtype, device=device)
    for j in range(p):
        prefix_table = prefix_coeffs[j]
        suffix_table = suffix_coeffs[j + 1]
        suffix_table_len = suffix_table.shape[1]

        d = torch.zeros((batch_size, p), dtype=dtype, device=device)
        for i in range(prefix_table.shape[1]):
            d[:, i : i + suffix_table_len] += prefix_table[:, i : i + 1] * suffix_table

        total_scale = torch.exp(prefix_log_scales[j] + suffix_log_scales[j + 1])
        out[j, :] = (delta[:, j] * (d @ w_in[1:]) + gamma[:, j] * (d @ w_out[:-1])) * total_scale

    return out.contiguous()


def _shapley_weights_signed(
    p: int, dtype: torch.dtype, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed Shapley size weights incl. the 1/p factor (w_out negative)."""
    w_in = torch.zeros(p + 1, dtype=dtype, device=device)
    w_out = torch.zeros(p + 1, dtype=dtype, device=device)
    for k in range(1, p + 1):
        w_in[k] = 1.0 / (math.comb(p - 1, k - 1) * p)
    for k in range(p):
        w_out[k] = -1.0 / (math.comb(p - 1, k) * p)
    return w_in, w_out


# ---------------------------------------------------------------------------
# ESP-accelerated A · K(Z, Z) · Aᵀ
# ---------------------------------------------------------------------------


def _shapley_w_in_out(p: int, dtype: torch.dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Unsigned Shapley size weights without the :math:`1/p` factor (applied as :math:`1/p^2` outside)."""
    w_in = torch.zeros(p + 1, dtype=dtype, device=device)
    w_out = torch.zeros(p + 1, dtype=dtype, device=device)
    for a in range(1, p + 1):
        w_in[a] = 1.0 / math.comb(p - 1, a - 1)
    for a in range(p):
        w_out[a] = 1.0 / math.comb(p - 1, a)
    return w_in, w_out


@torch.no_grad()
def akzza(lengthscales: torch.Tensor, outputscale: torch.Tensor) -> torch.Tensor:
    r"""Compute :math:`A K(Z, Z) A^\top` (:math:`p \times p`) via bivariate generating polynomials.

    For each player pair :math:`(i, j)` the double sum over coalition pairs
    collapses into a contraction of prefix/suffix coefficient tables of the
    bivariate generating polynomial
    :math:`\prod_r (\alpha_r + \beta_r \zeta_1 + \beta_r \zeta_2 + \alpha_r \zeta_1\zeta_2)`;
    tables are max-normalized per step with log-scale tracking for numerical
    stability.
    """
    alpha, beta = kernel_alpha_beta(lengthscales, outputscale)
    p = alpha.numel()
    dtype = alpha.dtype
    device = alpha.device
    tiny = torch.finfo(dtype).tiny

    w_in, w_out = _shapley_w_in_out(p, dtype=dtype, device=device)

    # Shifted weight vectors contracted into the tables' trailing dimension.
    O0 = w_out.clone()
    O1 = torch.zeros(p + 1, dtype=dtype, device=device)
    O1[:p] = w_out[1:]
    I1 = torch.zeros(p + 1, dtype=dtype, device=device)
    I1[:p] = w_in[1:]
    I2 = torch.zeros(p + 1, dtype=dtype, device=device)
    I2[: p - 1] = w_in[2:]

    LEFT_T = torch.stack([O0, O1, I1, I2]).T.contiguous()  # (p+1, 4)
    RIGHT_T = torch.stack([O0, O1, I1, I2]).T.contiguous()  # (p+1, 4)

    ps = p + 1
    alpha_cpu = alpha.tolist()
    beta_cpu = beta.tolist()

    P_left_curr = torch.zeros((ps, ps, 4), dtype=dtype, device=device)
    P_left_new = torch.empty_like(P_left_curr)
    suffix_contr_i = torch.zeros((p, ps, ps, 4), dtype=dtype, device=device)
    suffix_log_i = torch.zeros(p, dtype=dtype, device=device)
    S_right_prev = torch.zeros((ps, ps, 4), dtype=dtype, device=device)
    S_right_new = torch.empty_like(S_right_prev)

    AKZZA = torch.zeros((p, p), dtype=dtype, device=device)

    for i in range(p):
        factors_wo_i = [r for r in range(p) if r != i]
        nf = len(factors_wo_i)

        S_right_prev.zero_()
        S_right_new.zero_()
        suffix_contr_i.zero_()
        suffix_log_i.zero_()

        S_right_prev[0, :, :] = RIGHT_T
        log_S = 0.0

        # Backward pass: contracted suffix tables for all split points.
        for k in range(nf - 1, -1, -1):
            r = factors_wo_i[k]
            a_j, b_j = alpha[r], beta[r]

            S_right_new = a_j * S_right_prev
            S_right_new[1:, :, :] += b_j * S_right_prev[:-1, :, :]
            S_right_new[:, :-1, :] += b_j * S_right_prev[:, 1:, :]
            S_right_new[1:, :-1, :] += a_j * S_right_prev[:-1, 1:, :]

            s_new = S_right_new.abs().max().clamp_min(tiny)
            S_right_new = S_right_new / s_new
            log_S = log_S + s_new.log().item()

            suffix_contr_i[k] = S_right_new
            suffix_log_i[k] = log_S
            S_right_prev = S_right_new

        # Diagonal entry (i, i).
        P_left_curr.zero_()
        P_left_curr[:, 0, :] = LEFT_T
        log_P = 0.0

        contracted_diag = torch.einsum("abl,abr->lr", P_left_curr, suffix_contr_i[0])
        scale_diag = log_P + suffix_log_i[0].item()

        ai_v = alpha_cpu[i]
        bi_v = beta_cpu[i]
        diag_val = ai_v * (contracted_diag[2, 2] + contracted_diag[0, 0]) - bi_v * (
            contracted_diag[2, 0] + contracted_diag[0, 2]
        )
        AKZZA[i, i] = diag_val.item() * math.exp(scale_diag) / (p * p)

        # Forward pass: off-diagonal entries (i, j) for j < i (symmetric fill).
        for m in range(nf):
            j = factors_wo_i[m]

            if j > i:
                continue

            aj_v = alpha_cpu[j]
            bj_v = beta_cpu[j]

            if m < nf - 1:
                s_idx = m + 1
                contracted = torch.einsum("abl,abr->lr", P_left_curr, suffix_contr_i[s_idx])
                scale = log_P + suffix_log_i[s_idx].item()
            else:
                contracted = torch.einsum("bl,br->lr", P_left_curr[0, :, :], RIGHT_T)
                scale = log_P

            val = (
                (
                    ai_v * aj_v * contracted[0, 0]
                    + bi_v * aj_v * contracted[0, 1]
                    - ai_v * bj_v * contracted[0, 2]
                    - bi_v * bj_v * contracted[0, 3]
                    + ai_v * bj_v * contracted[1, 0]
                    + bi_v * bj_v * contracted[1, 1]
                    - ai_v * aj_v * contracted[1, 2]
                    - bi_v * aj_v * contracted[1, 3]
                    - bi_v * aj_v * contracted[2, 0]
                    - ai_v * aj_v * contracted[2, 1]
                    + bi_v * bj_v * contracted[2, 2]
                    + ai_v * bj_v * contracted[2, 3]
                    - bi_v * bj_v * contracted[3, 0]
                    - ai_v * bj_v * contracted[3, 1]
                    + bi_v * aj_v * contracted[3, 2]
                    + ai_v * aj_v * contracted[3, 3]
                ).item()
                * math.exp(scale)
                / (p * p)
            )

            AKZZA[i, j] = val
            AKZZA[j, i] = val

            a_j, b_j = alpha[j], beta[j]
            P_left_new = a_j * P_left_curr
            P_left_new[:-1, :, :] += b_j * P_left_curr[1:, :, :]
            P_left_new[:, 1:, :] += b_j * P_left_curr[:, :-1, :]
            P_left_new[:-1, 1:, :] += a_j * P_left_curr[1:, :-1, :]

            s_new = P_left_new.abs().max().clamp_min(tiny)
            P_left_new = P_left_new / s_new
            log_P = log_P + s_new.log().item()
            P_left_curr = P_left_new

    return AKZZA


def psd_chol(M: torch.Tensor) -> torch.Tensor:
    """Jitter-escalating Cholesky; identical routine to the legacy implementation."""
    return psd_safe_cholesky(M)


#: Backwards-compatible private alias (the reference implementation uses it).
_psd_chol = psd_chol


def noisy_train_kernel(gp_tensors: GPTensors) -> torch.Tensor:
    r"""Noisy train kernel :math:`K(X, X) + \sigma^2 I` on the training coalitions (standardized space)."""
    K = hamming_kernel(
        gp_tensors.train_X,
        gp_tensors.train_X,
        gp_tensors.lengthscales,
        gp_tensors.outputscale,
    )
    return K + gp_tensors.noise * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)


def a_sigma_w(
    A_KZW: torch.Tensor,
    A_KZX: torch.Tensor,
    K_XX_noisy_chol: torch.Tensor,
    K_XW: torch.Tensor,
) -> torch.Tensor:
    r"""Unscaled :math:`A \Sigma_\nu 1_W = A\_KZW - A\_KZX\,(K_{XX} + \sigma^2 I)^{-1} K_{XW}`.

    The second term is evaluated as ``(K_XW.T @ sol).T`` -- the same accumulation
    order linear_operator's lazy matmul uses -- for bit-parity with the legacy
    implementation.
    """
    sol = torch.cholesky_solve(A_KZX.T, K_XX_noisy_chol, upper=False)
    return A_KZW - (K_XW.mT @ sol).mT


def aea(AKA: torch.Tensor, A_KZX: torch.Tensor, K_XX_noisy_chol: torch.Tensor) -> torch.Tensor:
    r"""Unscaled :math:`A \Sigma_\nu A^\top`: :math:`AKZZA - A\_KZX\,(K+\sigma^2 I)^{-1} A\_KZX^\top`, symmetrized."""
    Y = torch.linalg.solve_triangular(K_XX_noisy_chol, A_KZX.T, upper=False)
    out = AKA - Y.T @ Y
    return 0.5 * (out + out.T)


def affine_posterior_mean(
    A_KZX: torch.Tensor, gp_tensors: GPTensors, K_XX_noisy_chol: torch.Tensor
) -> torch.Tensor:
    r"""Posterior property mean :math:`A \mu_\nu` in the original output scale.

    The constant-mean / row-sum-zero structure of :math:`A` makes all
    prior-mean terms vanish, leaving
    :math:`A\_KZX\,(K+\sigma^2 I)^{-1}\,(y_\mathrm{std} - m)` rescaled by the
    empirical std.
    """
    resid = (gp_tensors.train_y_std - gp_tensors.mean_const).reshape(-1, 1)
    sol = torch.cholesky_solve(resid, K_XX_noisy_chol, upper=False).reshape(-1)
    return (A_KZX @ sol) * gp_tensors.emp_std


def noisy_posterior_variance(
    W: torch.Tensor, gp_tensors: GPTensors, K_XX_noisy_chol: torch.Tensor
) -> torch.Tensor:
    """Diag Var[y(w)] at candidate coalitions, in the original output scale."""
    K_XW = hamming_kernel(
        gp_tensors.train_X,
        W.to(gp_tensors.train_X.dtype),
        gp_tensors.lengthscales,
        gp_tensors.outputscale,
    )
    Y = torch.linalg.solve_triangular(K_XX_noisy_chol, K_XW, upper=False)
    prior_diag = gp_tensors.outputscale + gp_tensors.noise  # k(w, w) = outputscale
    var_std = prior_diag - Y.square().sum(dim=0)
    return var_std * gp_tensors.emp_std**2


def shapleig_utilities(
    AEA_unscaled: torch.Tensor,
    B_unscaled: torch.Tensor,
    noisy_var_diag: torch.Tensor,
    emp_std: torch.Tensor,
) -> torch.Tensor:
    r"""Closed-form GOODE EIG per candidate: :math:`-\log(1 - q^\top (A\Sigma A^\top)^{-1} q / \mathrm{Var}[y])`."""
    L = _psd_chol(AEA_unscaled)
    Y = torch.linalg.solve_triangular(L, B_unscaled, upper=False)
    Q_diag = Y.square().sum(dim=0) * emp_std**2
    R = (Q_diag / noisy_var_diag.clamp_min(1e-30)).clamp(min=0.0, max=1.0 - 1e-12)
    return -torch.log1p(-R)


# ---------------------------------------------------------------------------
# Explicit cache for the incremental no-refit fast path
# ---------------------------------------------------------------------------


@dataclass
class ShaplEIGCache:
    """State reused across BED iterations when GP hyperparameters are frozen.

    Replaces the hidden ``object.__setattr__`` mutation cluster of the legacy
    implementation: every recycled quantity is an explicit field here.
    """

    A_KZX: torch.Tensor | None = None  # (p, t)   A · K(Z, X_train)
    A_KZW: torch.Tensor | None = None  # (p, S)   A · K(Z, candidates)
    AKA: torch.Tensor | None = None  # (p, p)   A · K(Z,Z) · Aᵀ (HP-only)
    AEA_unscaled: torch.Tensor | None = None  # (p, p) property covariance (unscaled)
    K_chol: torch.Tensor | None = None  # (t, t)  chol(K(X,X) + σ²I), current data/HPs
    candidates: torch.Tensor | None = None  # (S, p) candidate set of the last call
    num_candidates: int | None = None

    def reset(self) -> None:
        """Invalidate all cached quantities."""
        self.A_KZX = None
        self.A_KZW = None
        self.AKA = None
        self.AEA_unscaled = None
        self.K_chol = None
        self.candidates = None
        self.num_candidates = None

    def append_train_column(self, new_col: torch.Tensor) -> None:
        """Append the A.K(Z,x) column of a newly queried point to A_KZX."""
        assert self.A_KZX is not None
        self.A_KZX = torch.cat([self.A_KZX, new_col.reshape(-1, 1)], dim=1)

    def drop_candidate_column(self, idx: int) -> None:
        """Remove a selected candidate column from A_KZW."""
        assert self.A_KZW is not None
        self.A_KZW = torch.cat([self.A_KZW[:, :idx], self.A_KZW[:, idx + 1 :]], dim=1)
