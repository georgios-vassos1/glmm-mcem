"""Independence Metropolis-Hastings sampler for one subject's random intercept.

For a single subject *i* the target density is proportional to

    exp( log p(Y_i | u, beta) + log p(u | Vu) )

The *independence* MH proposal is a fixed Gaussian N(u_c, V_c) derived from
the Laplace approximation (see :mod:`glmm_mcem.laplace`).  Because the
proposal does not depend on the current state, the Hastings ratio simplifies
to

    r = exp( g(u*) - q(u*) ) / exp( g(u) - q(u) )

where *g* is the log-posterior and *q* is the log-proposal density.

For a symmetric Gaussian proposal this becomes

    r = exp( g(u*) - g(u) - log q(u*) + log q(u) )

which we evaluate on the log scale for numerical stability.

Functions
---------
sample_random_intercept
    Run *n_samples* steps of the chain for subject *i*, starting from
    ``current``, and return all retained samples (after burn-in).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .likelihoods import log_posterior


def sample_random_intercept(
    y_i: np.ndarray,
    X_i: np.ndarray,
    beta: np.ndarray,
    Vu: float,
    proposal_mean: float,
    proposal_variance: float,
    n_samples: int,
    burn_in: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Independence Metropolis-Hastings chain for one subject.

    Parameters
    ----------
    y_i : ndarray of shape (n_i,)
        Binary responses for subject *i*.
    X_i : ndarray of shape (n_i, p)
        Design-matrix rows for subject *i*.
    beta : ndarray of shape (p,)
        Current fixed-effect estimates.
    Vu : float
        Current random-effect variance.
    proposal_mean : float
        Mean of the (fixed) Gaussian proposal, typically the Laplace mode.
    proposal_variance : float
        Variance of the Gaussian proposal.
    n_samples : int
        Number of post-burn-in samples to return.
    burn_in : int, optional
        Number of initial steps to discard.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    samples : ndarray of shape (n_samples,)
        MCMC samples from the target posterior.
    """
    if rng is None:
        rng = np.random.default_rng()

    proposal_sd = np.sqrt(proposal_variance)
    total_steps = burn_in + n_samples

    # Pre-draw all proposals and uniform variates.
    candidates = rng.normal(loc=proposal_mean, scale=proposal_sd, size=total_steps)
    log_u      = np.log(rng.uniform(size=total_steps))

    # Log-density of the (fixed) proposal distribution.
    log_q = norm.logpdf  # partial applied below

    current         = proposal_mean  # start at proposal centre
    log_target_curr = log_posterior(current, y_i, X_i, beta, Vu)
    log_q_curr      = log_q(current, loc=proposal_mean, scale=proposal_sd)

    retained = np.empty(n_samples)
    keep_idx = 0

    for step in range(total_steps):
        candidate       = candidates[step]
        log_target_prop = log_posterior(candidate, y_i, X_i, beta, Vu)
        log_q_prop      = log_q(candidate, loc=proposal_mean, scale=proposal_sd)

        # Log acceptance ratio: log[ p(u*)/q(u*) ] - log[ p(u)/q(u) ]
        log_alpha = (log_target_prop - log_q_prop) - (log_target_curr - log_q_curr)

        if log_u[step] < log_alpha:
            current         = candidate
            log_target_curr = log_target_prop
            log_q_curr      = log_q_prop

        if step >= burn_in:
            retained[keep_idx] = current
            keep_idx += 1

    return retained
