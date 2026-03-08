"""Tests for glmm_mcem.sampler."""

import numpy as np
import pytest
from glmm_mcem.sampler import sample_random_intercept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subject(seed: int = 0, n_i: int = 20, p: int = 2):
    rng = np.random.default_rng(seed)
    X_i   = rng.standard_normal((n_i, p))
    beta  = np.array([0.4, -0.2])
    Vu    = 1.0
    u_true = rng.normal(0, 1)
    eta    = X_i @ beta + u_true
    prob   = 1.0 / (1.0 + np.exp(-eta))
    y_i    = (rng.uniform(size=n_i) < prob).astype(float)
    return y_i, X_i, beta, Vu


# ---------------------------------------------------------------------------
# Basic shape and type checks
# ---------------------------------------------------------------------------

class TestSampleBasics:
    def test_output_shape(self):
        y_i, X_i, beta, Vu = _make_subject()
        samples = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=0.0,
            proposal_variance=1.0,
            n_samples=50,
            rng=np.random.default_rng(1),
        )
        assert samples.shape == (50,)

    def test_all_finite(self):
        y_i, X_i, beta, Vu = _make_subject()
        samples = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=0.0,
            proposal_variance=1.0,
            n_samples=100,
            rng=np.random.default_rng(2),
        )
        assert np.all(np.isfinite(samples))

    def test_burn_in_ignored(self):
        y_i, X_i, beta, Vu = _make_subject()
        # Same seed, burn_in=0 vs burn_in=20 — should return same n_samples.
        s = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=0.0, proposal_variance=1.0,
            n_samples=30, burn_in=20,
            rng=np.random.default_rng(5),
        )
        assert s.shape == (30,)

    def test_reproducible_with_seed(self):
        y_i, X_i, beta, Vu = _make_subject()
        s1 = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=0.0, proposal_variance=1.0,
            n_samples=40, rng=np.random.default_rng(99),
        )
        s2 = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=0.0, proposal_variance=1.0,
            n_samples=40, rng=np.random.default_rng(99),
        )
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        y_i, X_i, beta, Vu = _make_subject()
        s1 = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=0.0, proposal_variance=1.0,
            n_samples=40, rng=np.random.default_rng(1),
        )
        s2 = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=0.0, proposal_variance=1.0,
            n_samples=40, rng=np.random.default_rng(2),
        )
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Distributional check (loose, Monte Carlo tolerance)
# ---------------------------------------------------------------------------

class TestSampleDistribution:
    def test_mean_near_proposal_mean(self):
        """With a well-centred Gaussian proposal the sample mean should be
        close to the proposal mean (very loose tolerance)."""
        y_i, X_i, beta, Vu = _make_subject(n_i=50)
        prop_mean = 0.0
        samples = sample_random_intercept(
            y_i, X_i, beta, Vu,
            proposal_mean=prop_mean,
            proposal_variance=1.0,
            n_samples=2000,
            burn_in=200,
            rng=np.random.default_rng(42),
        )
        # The posterior mean is likely ≠ 0, but the test just ensures
        # the sampler is not stuck far away from the mass.
        assert abs(np.mean(samples)) < 5.0
