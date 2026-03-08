"""Likelihood functions for logistic GLMMs with a normal random intercept.

Model
-----
For subject *i* with observations indexed by rows in *y_i*, *X_i*:

    eta_ij   = X_ij @ beta + u_i
    P(Y=1)   = sigmoid(eta_ij)
    log p(Y_i | u_i, beta)  = sum_j  [ y_ij * eta_ij - log1p(exp(eta_ij)) ]
    log p(u_i | Vu)          = -0.5 * ( log(2*pi*Vu) + u_i^2 / Vu )

The *log-posterior* of u_i given (Y_i, beta, Vu) is the sum of the above.

All functions are *scalar in u* so they can be passed directly to
:func:`scipy.optimize.minimize_scalar` or evaluated on 1-D grids.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def sigmoid(eta: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid :math:`1 / (1 + e^{-\\eta})`."""
    return np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))


# ---------------------------------------------------------------------------
# Per-subject log-likelihoods
# ---------------------------------------------------------------------------

def log_likelihood(
    u: float,
    y_i: np.ndarray,
    X_i: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Log-likelihood of *Y_i* given random intercept *u* and *beta*.

    Parameters
    ----------
    u : float
        Value of the random intercept for subject *i*.
    y_i : ndarray of shape (n_i,)
        Binary responses for subject *i*.
    X_i : ndarray of shape (n_i, p)
        Design-matrix rows for subject *i*.
    beta : ndarray of shape (p,)
        Fixed-effect coefficient vector.

    Returns
    -------
    float
        :math:`\\sum_j \\bigl[ y_{ij} \\eta_{ij} - \\log(1 + e^{\\eta_{ij}}) \\bigr]`
        where :math:`\\eta_{ij} = X_{ij} @ \\beta + u`.
    """
    eta = X_i @ beta + u
    # Stable: log(1 + exp(eta)) = eta + log1p(exp(-eta)) for eta >= 0
    #                            = log1p(exp(eta))        for eta < 0
    log1p_exp = np.where(
        eta >= 0,
        eta + np.log1p(np.exp(-eta)),
        np.log1p(np.exp(eta)),
    )
    return float(np.sum(y_i * eta - log1p_exp))


def log_prior(u: float, Vu: float) -> float:
    """Log-density of the normal prior :math:`u \\sim N(0, V_u)`.

    Parameters
    ----------
    u : float
        Random intercept value.
    Vu : float
        Prior variance (must be positive).

    Returns
    -------
    float
        :math:`-\\tfrac{1}{2}(\\log(2\\pi V_u) + u^2 / V_u)`.
    """
    return float(-0.5 * (np.log(2.0 * np.pi * Vu) + u**2 / Vu))


def log_posterior(
    u: float,
    y_i: np.ndarray,
    X_i: np.ndarray,
    beta: np.ndarray,
    Vu: float,
) -> float:
    """Log-posterior of *u_i* given (*Y_i*, *beta*, *Vu*).

    This is the unnormalised target used both in the Laplace approximation
    and as the acceptance probability numerator in the MH sampler.

    Returns
    -------
    float
        ``log_likelihood(u, ...) + log_prior(u, Vu)``.
    """
    return log_likelihood(u, y_i, X_i, beta) + log_prior(u, Vu)


# ---------------------------------------------------------------------------
# Score and Fisher-information contributions (for the M-step)
# ---------------------------------------------------------------------------

def score_contribution(
    u: float,
    y_i: np.ndarray,
    X_i: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Score vector :math:`\\partial \\log L_i / \\partial \\beta` at a fixed *u*.

    Parameters
    ----------
    u : float
        Current value of the random intercept.
    y_i, X_i, beta : see :func:`log_likelihood`.

    Returns
    -------
    ndarray of shape (p,)
        :math:`X_i^T (y_i - \\mu_i)` where :math:`\\mu_i = \\sigma(X_i \\beta + u)`.
    """
    eta = X_i @ beta + u
    mu = sigmoid(eta)
    return X_i.T @ (y_i - mu)


def fisher_contribution(
    u: float,
    y_i: np.ndarray,
    X_i: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Fisher information :math:`X_i^T W_i X_i` at a fixed *u*.

    Parameters
    ----------
    u : float
        Current value of the random intercept.
    y_i, X_i, beta : see :func:`log_likelihood`.

    Returns
    -------
    ndarray of shape (p, p)
        :math:`X_i^T \\mathrm{diag}(\\mu_i (1 - \\mu_i)) X_i`.
    """
    eta = X_i @ beta + u
    mu = sigmoid(eta)
    w = mu * (1.0 - mu)
    return (X_i * w[:, np.newaxis]).T @ X_i
