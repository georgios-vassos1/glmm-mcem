"""Tests for glmm_mcem.laplace."""

import numpy as np
import pytest
from glmm_mcem.laplace import (
    find_posterior_mode,
    compute_proposal_variance,
    laplace_approximation,
)
from glmm_mcem.likelihoods import log_posterior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subject(seed: int = 0, n_i: int = 20, p: int = 2):
    rng = np.random.default_rng(seed)
    X_i  = rng.standard_normal((n_i, p))
    beta = np.array([0.5, -0.3])
    Vu   = 1.0
    u_true = rng.normal(0, np.sqrt(Vu))
    eta    = X_i @ beta + u_true
    prob   = 1.0 / (1.0 + np.exp(-eta))
    y_i    = (rng.uniform(size=n_i) < prob).astype(float)
    return y_i, X_i, beta, Vu


# ---------------------------------------------------------------------------
# find_posterior_mode
# ---------------------------------------------------------------------------

class TestFindPosteriorMode:
    def test_returns_scalar(self):
        y_i, X_i, beta, Vu = _make_subject()
        mode = find_posterior_mode(y_i, X_i, beta, Vu)
        assert np.isscalar(mode) or mode.ndim == 0

    def test_mode_is_local_maximum(self):
        """Posterior at mode should exceed neighbours."""
        y_i, X_i, beta, Vu = _make_subject()
        mode = find_posterior_mode(y_i, X_i, beta, Vu)
        g_mode  = log_posterior(mode,       y_i, X_i, beta, Vu)
        g_plus  = log_posterior(mode + 0.1, y_i, X_i, beta, Vu)
        g_minus = log_posterior(mode - 0.1, y_i, X_i, beta, Vu)
        assert g_mode >= g_plus
        assert g_mode >= g_minus

    def test_warm_start_shifts_search(self):
        y_i, X_i, beta, Vu = _make_subject()
        # With a very large warm_start and small radius the mode should still
        # be found within the bracket (or the bounded solver clips to the edge).
        mode = find_posterior_mode(y_i, X_i, beta, Vu, warm_start=5.0, search_radius=15.0)
        assert np.isfinite(mode)


# ---------------------------------------------------------------------------
# compute_proposal_variance
# ---------------------------------------------------------------------------

class TestComputeProposalVariance:
    def test_positive(self):
        y_i, X_i, beta, Vu = _make_subject()
        mode = find_posterior_mode(y_i, X_i, beta, Vu)
        var  = compute_proposal_variance(mode, y_i, X_i, beta, Vu)
        assert var > 0.0

    def test_finite(self):
        y_i, X_i, beta, Vu = _make_subject()
        mode = find_posterior_mode(y_i, X_i, beta, Vu)
        var  = compute_proposal_variance(mode, y_i, X_i, beta, Vu)
        assert np.isfinite(var)

    def test_floor(self):
        # A single-observation subject with near-zero curvature should still
        # return a positive variance thanks to the 1e-6 floor.
        y_i  = np.array([1.0])
        X_i  = np.array([[0.0, 0.0]])
        beta = np.array([0.0, 0.0])
        Vu   = 1e6  # very diffuse prior → very flat posterior
        mode = find_posterior_mode(y_i, X_i, beta, Vu)
        var  = compute_proposal_variance(mode, y_i, X_i, beta, Vu)
        assert var >= 1e-6


# ---------------------------------------------------------------------------
# laplace_approximation
# ---------------------------------------------------------------------------

class TestLaplaceApproximation:
    def _build_dataset(self, n_subjects=5, n_i=10, p=2, seed=42):
        rng = np.random.default_rng(seed)
        beta = np.array([0.3, -0.5])
        Vu   = 1.0
        rows_y, rows_X, s_index = [], [], {}
        offset = 0
        for s in range(n_subjects):
            X_i = rng.standard_normal((n_i, p))
            u_i = rng.normal(0, 1)
            eta = X_i @ beta + u_i
            prob = 1.0 / (1.0 + np.exp(-eta))
            y_i  = (rng.uniform(size=n_i) < prob).astype(float)
            rows_y.append(y_i)
            rows_X.append(X_i)
            s_index[s] = list(range(offset, offset + n_i))
            offset += n_i
        return (
            np.concatenate(rows_y),
            np.vstack(rows_X),
            s_index,
            beta,
            Vu,
        )

    def test_keys_match_subjects(self):
        y, X, s_index, beta, Vu = self._build_dataset()
        proposals = laplace_approximation(y, X, s_index, beta, Vu)
        assert set(proposals.keys()) == set(s_index.keys())

    def test_variances_positive(self):
        y, X, s_index, beta, Vu = self._build_dataset()
        proposals = laplace_approximation(y, X, s_index, beta, Vu)
        for _, (mode, var) in proposals.items():
            assert var > 0.0

    def test_warm_starts_accepted(self):
        y, X, s_index, beta, Vu = self._build_dataset()
        warm = {s: 0.5 for s in s_index}
        proposals = laplace_approximation(y, X, s_index, beta, Vu, warm_starts=warm)
        assert len(proposals) == len(s_index)
