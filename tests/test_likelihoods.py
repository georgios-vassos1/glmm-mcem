"""Tests for glmm_mcem.likelihoods."""

import numpy as np
import pytest
from glmm_mcem.likelihoods import (
    sigmoid,
    log_likelihood,
    log_prior,
    log_posterior,
    score_contribution,
    fisher_contribution,
)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive(self):
        # Should be close to 1 without overflow.
        assert sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self):
        # Should be close to 0 without overflow.
        assert sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        x = np.linspace(-5, 5, 21)
        np.testing.assert_allclose(sigmoid(x) + sigmoid(-x), 1.0)

    def test_vector(self):
        x = np.array([-1.0, 0.0, 1.0])
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(sigmoid(x), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.n = 10
        self.p = 3
        self.X = rng.standard_normal((self.n, self.p))
        self.beta = np.array([0.5, -0.3, 0.8])
        self.y = rng.integers(0, 2, self.n).astype(float)

    def test_returns_finite_scalar(self):
        val = log_likelihood(0.0, self.y, self.X, self.beta)
        assert np.isfinite(val)

    def test_all_ones_response(self):
        # With y=1 everywhere and a very positive linear predictor,
        # the log-likelihood should be close to 0.
        X1 = np.ones((5, 1))
        beta1 = np.array([10.0])
        val = log_likelihood(0.0, np.ones(5), X1, beta1)
        assert val > -0.01

    def test_decreases_for_wrong_prediction(self):
        X1 = np.ones((5, 1))
        beta1 = np.array([10.0])
        # Predict y=1 strongly but truth is y=0: should be very negative.
        val = log_likelihood(0.0, np.zeros(5), X1, beta1)
        assert val < -40.0

    def test_against_manual_formula(self):
        u = 1.0
        eta = self.X @ self.beta + u
        log1p_exp = np.where(
            eta >= 0,
            eta + np.log1p(np.exp(-eta)),
            np.log1p(np.exp(eta)),
        )
        expected = float(np.sum(self.y * eta - log1p_exp))
        assert log_likelihood(u, self.y, self.X, self.beta) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# log_prior
# ---------------------------------------------------------------------------

class TestLogPrior:
    def test_mode_at_zero(self):
        # log-prior should be maximised at u=0.
        v = log_prior(0.0, 1.0)
        v_off = log_prior(0.5, 1.0)
        assert v > v_off

    def test_known_value(self):
        # N(0, 1): log p(0) = -0.5 * log(2*pi)
        expected = -0.5 * np.log(2.0 * np.pi)
        assert log_prior(0.0, 1.0) == pytest.approx(expected)

    def test_larger_variance_lower_penalty(self):
        # A diffuse prior penalises large |u| less once the quadratic term
        # dominates the log-normalisation.  The crossover for Vu=100 vs Vu=1
        # is at |u| ≈ sqrt(log(100)/0.99) ≈ 2.16, so use u=5.
        assert log_prior(5.0, 100.0) > log_prior(5.0, 1.0)


# ---------------------------------------------------------------------------
# log_posterior
# ---------------------------------------------------------------------------

class TestLogPosterior:
    def test_equals_sum(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((8, 2))
        beta = np.array([0.2, -0.5])
        y = rng.integers(0, 2, 8).astype(float)
        u, Vu = 0.7, 2.0
        expected = log_likelihood(u, y, X, beta) + log_prior(u, Vu)
        assert log_posterior(u, y, X, beta, Vu) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# score_contribution
# ---------------------------------------------------------------------------

class TestScoreContribution:
    def test_shape(self):
        rng = np.random.default_rng(2)
        p = 4
        X = rng.standard_normal((6, p))
        beta = rng.standard_normal(p)
        y = rng.integers(0, 2, 6).astype(float)
        s = score_contribution(0.0, y, X, beta)
        assert s.shape == (p,)

    def test_numerical_gradient(self):
        """Compare analytic score to finite-difference approximation."""
        rng = np.random.default_rng(3)
        p = 3
        X = rng.standard_normal((10, p))
        beta = np.array([0.1, -0.2, 0.3])
        y = rng.integers(0, 2, 10).astype(float)
        u = 0.5
        h = 1e-5

        analytic = score_contribution(u, y, X, beta)
        numeric  = np.zeros(p)
        for j in range(p):
            b_plus  = beta.copy(); b_plus[j]  += h
            b_minus = beta.copy(); b_minus[j] -= h
            numeric[j] = (
                log_likelihood(u, y, X, b_plus) - log_likelihood(u, y, X, b_minus)
            ) / (2 * h)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-4)


# ---------------------------------------------------------------------------
# fisher_contribution
# ---------------------------------------------------------------------------

class TestFisherContribution:
    def test_shape(self):
        rng = np.random.default_rng(4)
        p = 3
        X = rng.standard_normal((5, p))
        beta = rng.standard_normal(p)
        y = rng.integers(0, 2, 5).astype(float)
        F = fisher_contribution(0.0, y, X, beta)
        assert F.shape == (p, p)

    def test_positive_semidefinite(self):
        rng = np.random.default_rng(5)
        p = 3
        X = rng.standard_normal((20, p))
        beta = np.zeros(p)
        y = rng.integers(0, 2, 20).astype(float)
        F = fisher_contribution(0.0, y, X, beta)
        eigenvalues = np.linalg.eigvalsh(F)
        assert np.all(eigenvalues >= -1e-12)
