"""Simulation study replicating the R reference from Baolin Wu's teaching page.

Setup
-----
* n = 1 000 subjects
* K = 4 observations per subject  (balanced design)
* p = 3 predictors  (intercept + 2 covariates)
* True beta = [-1, 1, -1]
* True Vu   = 100  (large random effect)

The script fits the model with MCEM, prints parameter estimates alongside
the truth, and reports the final convergence status.

Usage::

    uv run python demo/simulation_study.py

Expected output (approximate)::

    ============================================================
    GLMM-MCEM Simulation Study
    n_subjects=1000  K=4  p=3  Vu_true=100
    ============================================================

    Fitting model ...

    ============================================================
    Results
    ============================================================
    Parameter   True     Estimate
    ----------  -------  --------
    beta[0]     -1.0000    -x.xxxx
    beta[1]      1.0000     x.xxxx
    beta[2]     -1.0000    -x.xxxx
    Vu          100.0000   xx.xxxx

    Converged : True  (N iterations: xx)
    ============================================================
"""

from __future__ import annotations

import time

import numpy as np

from glmm_mcem import build_dataset, run_mcem


# ---------------------------------------------------------------------------
# Simulation parameters (match R reference)
# ---------------------------------------------------------------------------

N_SUBJECTS     = 1_000
OBS_PER_SUBJ   = 4        # K
BETA_TRUE      = np.array([-1.0, 1.0, -1.0])
VU_TRUE        = 100.0
SEED           = 2024

# MCEM hyper-parameters
N_ITERATIONS   = 30
N_MC_SAMPLES   = 200
BURN_IN        = 100
STEP_SIZE      = 0.5      # dampened NR for large Vu
INITIAL_VAR    = 100.0    # warm-start Vu near the truth


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def simulate_glmm(
    n_subjects: int,
    obs_per_subj: int,
    beta_true: np.ndarray,
    Vu_true: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    p   = len(beta_true)
    n_obs = n_subjects * obs_per_subj

    X = np.column_stack([
        np.ones(n_obs),                          # intercept
        rng.standard_normal((n_obs, p - 1)),     # covariates
    ])
    subject_ids = np.repeat(np.arange(n_subjects), obs_per_subj)
    u = rng.normal(0.0, np.sqrt(Vu_true), size=n_subjects)
    u_expanded = u[subject_ids]

    eta  = X @ beta_true + u_expanded
    prob = 1.0 / (1.0 + np.exp(-eta))
    y    = (rng.uniform(size=n_obs) < prob).astype(float)

    return build_dataset(y, X, subject_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GLMM-MCEM Simulation Study")
    print(
        f"n_subjects={N_SUBJECTS}  K={OBS_PER_SUBJ}"
        f"  p={len(BETA_TRUE)}  Vu_true={VU_TRUE}"
    )
    print("=" * 60)

    data = simulate_glmm(
        n_subjects=N_SUBJECTS,
        obs_per_subj=OBS_PER_SUBJ,
        beta_true=BETA_TRUE,
        Vu_true=VU_TRUE,
        seed=SEED,
    )
    print(f"\nDataset: {data.n_obs} observations, {data.n_subjects} subjects\n")

    print("Fitting model ...")
    t0 = time.perf_counter()
    result = run_mcem(
        data,
        initial_variance=INITIAL_VAR,
        n_iterations=N_ITERATIONS,
        n_mc_samples=N_MC_SAMPLES,
        burn_in=BURN_IN,
        step_size=STEP_SIZE,
        tol=1e-3,
        seed=SEED,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"{'Parameter':<12}{'True':>10}{'Estimate':>12}")
    print(f"{'----------':<12}{'-------':>10}{'--------':>12}")
    for j, (b_true, b_est) in enumerate(zip(BETA_TRUE, result.beta)):
        print(f"{'beta[' + str(j) + ']':<12}{b_true:>10.4f}{b_est:>12.4f}")
    print(f"{'Vu':<12}{VU_TRUE:>10.4f}{result.Vu:>12.4f}")
    print()
    print(f"Converged : {result.converged}  (N iterations: {result.n_iter})")
    print(f"Wall time : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
