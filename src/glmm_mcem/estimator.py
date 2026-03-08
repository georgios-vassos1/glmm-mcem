"""M-step estimators for the MCEM algorithm.

The E-step integrates out the random effects by Monte Carlo averaging.
This module provides the combined E/M-step function that:

1. Runs the MH sampler for every subject (E-step).
2. Averages score and Fisher-information contributions over the samples.
3. Performs one Newton-Raphson step to update *beta* (M-step for beta).
4. Updates *Vu* as the sample mean of :math:`u^2` across all subjects and
   Monte Carlo draws (M-step for Vu).

The single-pass design (one function returning scores, Fisher info *and*
samples) avoids running the sampler twice per MCEM iteration.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .likelihoods import score_contribution, fisher_contribution
from .sampler import sample_random_intercept


def compute_mc_score_fisher_and_samples(
    data_y: np.ndarray,
    data_X: np.ndarray,
    subject_index: Dict[int, List[int]],
    beta: np.ndarray,
    Vu: float,
    proposals: Dict[int, Tuple[float, float]],
    n_mc_samples: int = 100,
    burn_in: int = 50,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """E-step: compute MC-averaged score and Fisher information.

    For each subject *i* the sampler draws :math:`u_i^{(1)}, \\ldots, u_i^{(m)}`
    from the MH chain; the score and Fisher information are then averaged:

    .. math::

        \\hat{S}   &= \\frac{1}{n} \\sum_i \\frac{1}{m} \\sum_k s(\\beta, u_i^{(k)}) \\\\
        \\hat{\\mathcal{I}} &= \\frac{1}{n} \\sum_i \\frac{1}{m} \\sum_k F(\\beta, u_i^{(k)})

    Parameters
    ----------
    data_y : ndarray of shape (n_obs,)
        Full binary response vector.
    data_X : ndarray of shape (n_obs, p)
        Full design matrix.
    subject_index : dict
        Mapping ``subject_id -> list[row_index]``.
    beta : ndarray of shape (p,)
        Current fixed-effect estimates.
    Vu : float
        Current random-effect variance.
    proposals : dict
        ``proposals[i] = (mean_i, variance_i)`` from the Laplace approximation.
    n_mc_samples : int, optional
        Number of Monte Carlo samples per subject.
    burn_in : int, optional
        Burn-in steps discarded from the MH chain.
    rng : numpy.random.Generator, optional
        Shared random-number generator (for reproducibility).

    Returns
    -------
    score : ndarray of shape (p,)
        MC-averaged score vector.
    fisher_info : ndarray of shape (p, p)
        MC-averaged Fisher information matrix.
    samples_by_subject : dict
        ``samples_by_subject[i]`` is an ndarray of shape ``(n_mc_samples,)``
        containing the MH samples for subject *i*.  Returned for reuse in the
        Vu update step.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = beta.shape[0]
    score_total      = np.zeros(p)
    fisher_total     = np.zeros((p, p))
    samples_by_subject: Dict[int, np.ndarray] = {}

    n_subjects = len(subject_index)

    for subj, rows in subject_index.items():
        y_i = data_y[rows]
        X_i = data_X[rows]
        prop_mean, prop_var = proposals[subj]

        samples = sample_random_intercept(
            y_i=y_i,
            X_i=X_i,
            beta=beta,
            Vu=Vu,
            proposal_mean=prop_mean,
            proposal_variance=prop_var,
            n_samples=n_mc_samples,
            burn_in=burn_in,
            rng=rng,
        )
        samples_by_subject[subj] = samples

        # Average score and Fisher contributions over MC samples.
        s_sum = np.zeros(p)
        F_sum = np.zeros((p, p))
        for u_draw in samples:
            s_sum += score_contribution(u_draw, y_i, X_i, beta)
            F_sum += fisher_contribution(u_draw, y_i, X_i, beta)

        score_total  += s_sum  / n_mc_samples
        fisher_total += F_sum  / n_mc_samples

    # Normalise by number of subjects (consistent with per-subject gradients).
    score_total  /= n_subjects
    fisher_total /= n_subjects

    return score_total, fisher_total, samples_by_subject


def newton_raphson_beta_update(
    beta: np.ndarray,
    score: np.ndarray,
    fisher_info: np.ndarray,
    step_size: float = 1.0,
    ridge: float = 1e-6,
) -> np.ndarray:
    """One Newton-Raphson step for *beta*.

    Solves ``(FI + ridge * I) d = score`` and returns ``beta + step_size * d``.

    The small ridge penalty improves numerical stability when the Fisher
    information matrix is nearly singular (few observations per subject,
    poorly identified models).

    Parameters
    ----------
    beta : ndarray of shape (p,)
        Current coefficient vector.
    score : ndarray of shape (p,)
        MC-averaged score vector.
    fisher_info : ndarray of shape (p, p)
        MC-averaged Fisher information matrix.
    step_size : float, optional
        Damping factor ∈ (0, 1] to control step length.
    ridge : float, optional
        Ridge regularisation for numerical stability.

    Returns
    -------
    beta_new : ndarray of shape (p,)
        Updated coefficient vector.
    """
    p = beta.shape[0]
    regularised = fisher_info + ridge * np.eye(p)
    direction   = np.linalg.solve(regularised, score)
    return beta + step_size * direction


def update_Vu(samples_by_subject: Dict[int, np.ndarray]) -> float:
    """Update *Vu* as the mean squared random effect across all MC samples.

    Under the normal prior :math:`u_i \\sim N(0, V_u)` the M-step estimator is

    .. math::

        \\hat{V}_u = \\frac{1}{n} \\sum_i \\frac{1}{m} \\sum_k (u_i^{(k)})^2

    Parameters
    ----------
    samples_by_subject : dict
        Output of :func:`compute_mc_score_fisher_and_samples`.

    Returns
    -------
    float
        New estimate of *Vu*, floored at ``1e-6`` to prevent degeneracy.
    """
    mean_sq = np.mean([np.mean(s**2) for s in samples_by_subject.values()])
    return float(max(mean_sq, 1e-6))
