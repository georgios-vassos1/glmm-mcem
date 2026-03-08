"""Outer MCEM loop for logistic GLMMs.

This module ties together the building blocks from the other sub-modules:

* :mod:`glmm_mcem.data`        — dataset container
* :mod:`glmm_mcem.laplace`     — Gaussian proposal construction
* :mod:`glmm_mcem.estimator`   — E/M-step computations

The entry point is :func:`run_mcem`.

Algorithm outline
-----------------
::

    0.  beta_0  = marginal logistic MLE  (cold start, ignores random effects)
        Vu_0    = initial_variance
        u_c_i   = 0, V_c_i = Vu_0  for all i

    1.  proposals <- laplace_approximation for all i

    For b in 1 .. n_iterations:
        score, FI, samples = compute_mc_score_fisher_and_samples(...)
        beta_new  = beta  + step_size * solve(FI + ridge*I, score)
        Vu_new    = mean_i( mean_k( u_i^(k)^2 ) )
        proposals = laplace_approximation(..., warm_starts=modes_from_proposals)
        if ||beta_new - beta|| < tol  and  |Vu_new - Vu| / Vu < tol:  break
        beta, Vu  = beta_new, Vu_new

    return McemResult(beta, Vu, history, converged)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .data import GlmmData
from .laplace import laplace_approximation
from .estimator import (
    compute_mc_score_fisher_and_samples,
    newton_raphson_beta_update,
    update_Vu,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class McemResult:
    """Return value of :func:`run_mcem`.

    Attributes
    ----------
    beta : ndarray of shape (p,)
        Final fixed-effect estimates.
    Vu : float
        Final random-effect variance estimate.
    history : list of dict
        One entry per completed iteration with keys
        ``{'iteration', 'beta', 'Vu', 'delta_beta', 'delta_Vu'}``.
    converged : bool
        ``True`` if the convergence criterion was met before *n_iterations*.
    n_iter : int
        Number of iterations actually performed.
    """

    beta: np.ndarray
    Vu: float
    history: List[dict] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


# ---------------------------------------------------------------------------
# Cold-start initialisation
# ---------------------------------------------------------------------------

def _marginal_logistic_mle(
    y: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Marginal logistic MLE ignoring the random intercept.

    Minimises the negative log-likelihood of the pooled logistic model via
    L-BFGS-B.  Used as a warm start for *beta* in :func:`run_mcem`.

    Parameters
    ----------
    y : ndarray of shape (n_obs,)
    X : ndarray of shape (n_obs, p)

    Returns
    -------
    ndarray of shape (p,)
        MLE estimate of *beta*.
    """
    p = X.shape[1]

    def neg_log_lik(b: np.ndarray) -> float:
        eta = X @ b
        log1p_exp = np.where(
            eta >= 0,
            eta + np.log1p(np.exp(-eta)),
            np.log1p(np.exp(eta)),
        )
        return float(np.sum(log1p_exp - y * eta))

    def grad(b: np.ndarray) -> np.ndarray:
        eta = X @ b
        mu  = np.where(
            eta >= 0,
            1.0 / (1.0 + np.exp(-eta)),
            np.exp(eta) / (1.0 + np.exp(eta)),
        )
        return X.T @ (mu - y)

    result = minimize(
        neg_log_lik,
        x0=np.zeros(p),
        jac=grad,
        method="L-BFGS-B",
    )
    return result.x


# ---------------------------------------------------------------------------
# Main MCEM loop
# ---------------------------------------------------------------------------

def run_mcem(
    data: GlmmData,
    *,
    initial_variance: float = 1.0,
    n_iterations: int = 50,
    n_mc_samples: int = 100,
    burn_in: int = 50,
    step_size: float = 1.0,
    ridge: float = 1e-6,
    tol: float = 1e-4,
    laplace_search_radius: float = 10.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> McemResult:
    """Fit a logistic GLMM by MCEM.

    Parameters
    ----------
    data : GlmmData
        Dataset prepared by :func:`~glmm_mcem.data.build_dataset`.
    initial_variance : float, optional
        Starting value for *Vu* (the random-effect variance).
    n_iterations : int, optional
        Maximum number of MCEM outer iterations.
    n_mc_samples : int, optional
        Number of MH samples per subject per iteration (the Monte Carlo
        sample size *m*; larger → lower MC variance but slower).
    burn_in : int, optional
        Number of MH steps to discard at the start of each chain.
    step_size : float, optional
        Damping for the Newton-Raphson *beta* update (0 < step_size ≤ 1).
    ridge : float, optional
        Ridge penalty added to the Fisher information for numerical stability.
    tol : float, optional
        Convergence threshold: stop when
        ``‖beta_new − beta‖ < tol  and  |Vu_new − Vu| / Vu < tol``.
    laplace_search_radius : float, optional
        Half-width of the bounded search interval in :func:`~glmm_mcem.laplace.find_posterior_mode`.
    seed : int, optional
        Seed for the random-number generator (for reproducibility).
    verbose : bool, optional
        If ``True``, print a one-line summary after each iteration.

    Returns
    -------
    McemResult
    """
    rng = np.random.default_rng(seed)

    # ---- Cold start ----
    beta = _marginal_logistic_mle(data.y, data.X)
    Vu   = float(initial_variance)

    if verbose:
        print(f"Cold-start  beta={np.round(beta, 4)}  Vu={Vu:.4f}")

    # ---- Initial Laplace proposals ----
    proposals = laplace_approximation(
        data_y=data.y,
        data_X=data.X,
        subject_index=data.subject_index,
        beta=beta,
        Vu=Vu,
        warm_starts=None,
        search_radius=laplace_search_radius,
    )

    history: List[dict] = []
    converged = False
    iteration = 0

    for iteration in range(1, n_iterations + 1):
        # ---- E-step + M-step ----
        score, fisher_info, samples = compute_mc_score_fisher_and_samples(
            data_y=data.y,
            data_X=data.X,
            subject_index=data.subject_index,
            beta=beta,
            Vu=Vu,
            proposals=proposals,
            n_mc_samples=n_mc_samples,
            burn_in=burn_in,
            rng=rng,
        )

        beta_new = newton_raphson_beta_update(
            beta=beta,
            score=score,
            fisher_info=fisher_info,
            step_size=step_size,
            ridge=ridge,
        )
        Vu_new = update_Vu(samples)

        # ---- Convergence check ----
        delta_beta = float(np.linalg.norm(beta_new - beta))
        delta_Vu   = abs(Vu_new - Vu) / max(Vu, 1e-10)

        history.append({
            "iteration": iteration,
            "beta": beta_new.copy(),
            "Vu": Vu_new,
            "delta_beta": delta_beta,
            "delta_Vu": delta_Vu,
        })

        if verbose:
            print(
                f"  iter {iteration:3d}"
                f"  beta={np.round(beta_new, 4)}"
                f"  Vu={Vu_new:.4f}"
                f"  Δbeta={delta_beta:.2e}"
                f"  ΔVu={delta_Vu:.2e}"
            )

        beta = beta_new
        Vu   = Vu_new

        # ---- Update proposals with warm starts from current modes ----
        warm_starts: Dict[int, float] = {
            subj: proposals[subj][0] for subj in proposals
        }
        proposals = laplace_approximation(
            data_y=data.y,
            data_X=data.X,
            subject_index=data.subject_index,
            beta=beta,
            Vu=Vu,
            warm_starts=warm_starts,
            search_radius=laplace_search_radius,
        )

        if delta_beta < tol and delta_Vu < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {iteration}.")
            break

    return McemResult(
        beta=beta,
        Vu=Vu,
        history=history,
        converged=converged,
        n_iter=iteration,
    )
