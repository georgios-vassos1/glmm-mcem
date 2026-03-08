"""Tests for glmm_mcem.mcem (and the high-level API)."""

from __future__ import annotations

import numpy as np
import pytest
from glmm_mcem import build_dataset, run_mcem, McemResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_data(
    n_subjects: int = 50,
    obs_per_subject: int = 4,
    beta_true: np.ndarray | None = None,
    Vu_true: float = 1.0,
    seed: int = 0,
):
    """Simulate a balanced logistic GLMM dataset."""
    if beta_true is None:
        beta_true = np.array([0.5, -0.3, 0.8])
    rng = np.random.default_rng(seed)
    p = len(beta_true)
    n_obs = n_subjects * obs_per_subject

    X = rng.standard_normal((n_obs, p))
    subject_ids = np.repeat(np.arange(n_subjects), obs_per_subject)
    u = rng.normal(0, np.sqrt(Vu_true), size=n_subjects)
    u_expanded = u[subject_ids]

    eta  = X @ beta_true + u_expanded
    prob = 1.0 / (1.0 + np.exp(-eta))
    y    = (rng.uniform(size=n_obs) < prob).astype(float)

    return build_dataset(y, X, subject_ids), beta_true, Vu_true


# ---------------------------------------------------------------------------
# McemResult
# ---------------------------------------------------------------------------

class TestMcemResult:
    def test_fields(self):
        result = McemResult(
            beta=np.array([1.0, 2.0]),
            Vu=0.5,
            history=[],
            converged=True,
            n_iter=3,
        )
        assert result.converged
        assert result.n_iter == 3
        np.testing.assert_array_equal(result.beta, [1.0, 2.0])


# ---------------------------------------------------------------------------
# run_mcem: smoke tests
# ---------------------------------------------------------------------------

class TestRunMcemSmoke:
    def test_returns_mcemresult(self):
        data, _, _ = _simulate_data(n_subjects=20, seed=7)
        result = run_mcem(data, n_iterations=3, n_mc_samples=20, burn_in=10, seed=0)
        assert isinstance(result, McemResult)

    def test_beta_shape(self):
        data, beta_true, _ = _simulate_data(n_subjects=20, seed=8)
        result = run_mcem(data, n_iterations=3, n_mc_samples=20, burn_in=10, seed=1)
        assert result.beta.shape == beta_true.shape

    def test_Vu_positive(self):
        data, _, _ = _simulate_data(n_subjects=20, seed=9)
        result = run_mcem(data, n_iterations=3, n_mc_samples=20, burn_in=10, seed=2)
        assert result.Vu > 0.0

    def test_history_length(self):
        data, _, _ = _simulate_data(n_subjects=10, seed=10)
        result = run_mcem(data, n_iterations=5, n_mc_samples=20, burn_in=10, seed=3)
        assert len(result.history) <= 5
        assert len(result.history) >= 1

    def test_history_keys(self):
        data, _, _ = _simulate_data(n_subjects=10, seed=11)
        result = run_mcem(data, n_iterations=3, n_mc_samples=20, burn_in=10, seed=4)
        for entry in result.history:
            assert "iteration" in entry
            assert "beta" in entry
            assert "Vu" in entry

    def test_verbose_runs(self, capsys):
        data, _, _ = _simulate_data(n_subjects=10, seed=12)
        run_mcem(data, n_iterations=2, n_mc_samples=10, burn_in=5, seed=5, verbose=True)
        captured = capsys.readouterr()
        assert "beta" in captured.out.lower() or "iter" in captured.out.lower()

    def test_seed_reproducibility(self):
        data, _, _ = _simulate_data(n_subjects=15, seed=13)
        r1 = run_mcem(data, n_iterations=3, n_mc_samples=20, burn_in=10, seed=99)
        r2 = run_mcem(data, n_iterations=3, n_mc_samples=20, burn_in=10, seed=99)
        np.testing.assert_array_equal(r1.beta, r2.beta)
        assert r1.Vu == r2.Vu


# ---------------------------------------------------------------------------
# run_mcem: estimation accuracy (loose checks on small sample)
# ---------------------------------------------------------------------------

class TestRunMcemEstimation:
    def test_beta_direction(self):
        """Check that the sign of each coefficient is recovered."""
        beta_true = np.array([1.0, -1.0, 0.5])
        data, _, _ = _simulate_data(
            n_subjects=40, obs_per_subject=8,
            beta_true=beta_true, Vu_true=1.0, seed=42,
        )
        result = run_mcem(
            data,
            n_iterations=8,
            n_mc_samples=40,
            burn_in=20,
            seed=0,
        )
        # Signs should agree with the truth.
        assert np.sign(result.beta[0]) == np.sign(beta_true[0])
        assert np.sign(result.beta[1]) == np.sign(beta_true[1])

    def test_Vu_in_plausible_range(self):
        """Vu estimate should be in a broad neighbourhood of the truth."""
        data, _, Vu_true = _simulate_data(
            n_subjects=40, obs_per_subject=8, Vu_true=1.0, seed=43,
        )
        result = run_mcem(
            data,
            n_iterations=8,
            n_mc_samples=40,
            burn_in=20,
            seed=0,
        )
        # Very loose: within a factor of 5 on either side.
        assert 0.1 < result.Vu < 10.0

    def test_convergence_flag_set(self):
        """With a very loose tolerance run_mcem should declare convergence."""
        rng = np.random.default_rng(0)
        X = np.ones((200, 1))
        y = rng.integers(0, 2, 200).astype(float)
        data = build_dataset(y, X, np.repeat(np.arange(50), 4))
        result = run_mcem(
            data,
            n_iterations=100,
            n_mc_samples=50,
            burn_in=20,
            tol=1.0,  # very loose — converges in a handful of iterations
            seed=0,
        )
        assert result.converged
        assert result.n_iter < 100
