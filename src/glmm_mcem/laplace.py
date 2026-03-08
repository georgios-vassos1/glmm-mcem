"""Laplace approximation for the per-subject random-intercept posterior.

For subject *i* the (unnormalised) log-posterior is

    g(u) = log p(Y_i | u, beta) + log p(u | Vu)

The Laplace approximation matches a Gaussian to the curvature at the mode:

    u*   = argmax_u  g(u)
    V*   = -1 / g''(u*)   (inverse of the negative curvature)

This Gaussian N(u*, V*) is used as the *proposal distribution* for the
independence Metropolis-Hastings sampler in :mod:`glmm_mcem.sampler`.

Functions
---------
find_posterior_mode
    Scalar bounded optimisation via :func:`scipy.optimize.minimize_scalar`.
compute_proposal_variance
    Central finite-difference second derivative evaluated at *u**.
laplace_approximation
    Convenience wrapper that runs both steps for all subjects.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .likelihoods import log_posterior


# ---------------------------------------------------------------------------
# Single-subject helpers
# ---------------------------------------------------------------------------

def find_posterior_mode(
    y_i: np.ndarray,
    X_i: np.ndarray,
    beta: np.ndarray,
    Vu: float,
    warm_start: float = 0.0,
    search_radius: float = 10.0,
) -> float:
    """Find the mode of :func:`~glmm_mcem.likelihoods.log_posterior` for one subject.

    Uses :func:`scipy.optimize.minimize_scalar` with ``method='bounded'``
    over the interval ``[warm_start - search_radius, warm_start + search_radius]``.

    Parameters
    ----------
    y_i : ndarray of shape (n_i,)
        Binary responses.
    X_i : ndarray of shape (n_i, p)
        Design rows.
    beta : ndarray of shape (p,)
        Current fixed-effect estimates.
    Vu : float
        Current random-effect variance.
    warm_start : float, optional
        Centre of the search bracket (previous mode estimate).
    search_radius : float, optional
        Half-width of the bounded search interval.

    Returns
    -------
    float
        The mode :math:`u^*`.
    """
    lo = warm_start - search_radius
    hi = warm_start + search_radius

    result = minimize_scalar(
        fun=lambda u: -log_posterior(u, y_i, X_i, beta, Vu),
        bounds=(lo, hi),
        method="bounded",
    )
    return float(result.x)


def compute_proposal_variance(
    mode: float,
    y_i: np.ndarray,
    X_i: np.ndarray,
    beta: np.ndarray,
    Vu: float,
    h: float = 1e-4,
) -> float:
    """Estimate proposal variance via central finite-difference curvature.

    Approximates the second derivative of :func:`~glmm_mcem.likelihoods.log_posterior`
    at *mode* and returns its negative reciprocal:

    .. math::

        V^* = -\\frac{1}{g''(u^*)}
            \\approx -\\frac{1}{(g(u^*+h) - 2g(u^*) + g(u^*-h)) / h^2}

    A floor of ``1e-6`` prevents non-positive variance when the posterior is
    very flat.

    Parameters
    ----------
    mode : float
        The posterior mode :math:`u^*`.
    y_i, X_i, beta, Vu : see :func:`find_posterior_mode`.
    h : float, optional
        Step size for the finite-difference approximation.

    Returns
    -------
    float
        Estimated proposal variance (positive).
    """
    g_plus  = log_posterior(mode + h, y_i, X_i, beta, Vu)
    g_mid   = log_posterior(mode,     y_i, X_i, beta, Vu)
    g_minus = log_posterior(mode - h, y_i, X_i, beta, Vu)

    second_deriv = (g_plus - 2.0 * g_mid + g_minus) / (h ** 2)
    # second_deriv should be <= 0 at a maximum; take absolute value for safety.
    variance = 1.0 / max(-second_deriv, 1e-6)
    return float(variance)


# ---------------------------------------------------------------------------
# Full-dataset interface
# ---------------------------------------------------------------------------

def laplace_approximation(
    data_y: np.ndarray,
    data_X: np.ndarray,
    subject_index: Dict[int, List[int]],
    beta: np.ndarray,
    Vu: float,
    warm_starts: Dict[int, float] | None = None,
    search_radius: float = 10.0,
) -> Dict[int, Tuple[float, float]]:
    """Run the Laplace approximation for every subject.

    Parameters
    ----------
    data_y : ndarray of shape (n_obs,)
        Full response vector.
    data_X : ndarray of shape (n_obs, p)
        Full design matrix.
    subject_index : dict
        Mapping ``subject_id -> list[row_index]`` (from :class:`~glmm_mcem.data.GlmmData`).
    beta : ndarray of shape (p,)
        Current fixed-effect estimates.
    Vu : float
        Current random-effect variance.
    warm_starts : dict, optional
        Previous mode estimates keyed by subject id.  Defaults to 0.0 for
        all subjects.
    search_radius : float, optional
        Passed to :func:`find_posterior_mode`.

    Returns
    -------
    proposals : dict
        ``proposals[i] = (mode_i, variance_i)`` for each subject *i*.
    """
    if warm_starts is None:
        warm_starts = {}

    proposals: Dict[int, Tuple[float, float]] = {}
    for subj, rows in subject_index.items():
        y_i = data_y[rows]
        X_i = data_X[rows]
        ws  = warm_starts.get(subj, 0.0)

        mode = find_posterior_mode(
            y_i, X_i, beta, Vu,
            warm_start=ws,
            search_radius=search_radius,
        )
        variance = compute_proposal_variance(mode, y_i, X_i, beta, Vu)
        proposals[subj] = (mode, variance)

    return proposals
