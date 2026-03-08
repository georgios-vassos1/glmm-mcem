# glmm-mcem

A clean-room Python implementation of **Monte Carlo Expectation-Maximisation (MCEM)** for logistic Generalised Linear Mixed Models (GLMMs) with a single normal random intercept.

The algorithm is based on Baolin Wu's teaching material and re-implements the original R code as a readable, modular, dependency-light Python library.

---

## The model

For subject $i$ with observations $j$:

$$
Y_{ij} \mid u_i \sim \text{Bernoulli}\!\left(\sigma(X_{ij}^\top \beta + u_i)\right), \qquad u_i \sim \mathcal{N}(0, V_u)
$$

**Parameters:** fixed-effect vector $\beta \in \mathbb{R}^p$ and random-effect variance $V_u > 0$.

The MCEM algorithm alternates between:
- **E-step** — sample $u_i$ from its posterior via an independence Metropolis-Hastings chain (Laplace-approximation proposal)
- **M-step** — update $\beta$ with Newton-Raphson and $V_u$ as the mean squared sample

---

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) (package manager)

Runtime dependencies are **numpy** and **scipy** only. `matplotlib` and `jupyter` are dev-only (needed for the tutorial notebook).

---

## Installation

```bash
git clone <repo-url>
cd glmm-mcem
make install          # uv sync: creates .venv and installs all deps
```

Or without `make`:

```bash
uv sync
```

---

## Quick start

```python
import numpy as np
from glmm_mcem import build_dataset, run_mcem

# 1. Prepare your data
#    y            — binary response vector (0/1), shape (n_obs,)
#    X            — fixed-effects design matrix, shape (n_obs, p)
#    subject_ids  — subject label per observation, shape (n_obs,)
data = build_dataset(y, X, subject_ids)

# 2. Fit
result = run_mcem(data, n_iterations=30, n_mc_samples=150, seed=0)

# 3. Read estimates
print(result.beta)       # ndarray (p,)
print(result.Vu)         # float
print(result.converged)  # bool
```

`run_mcem` key parameters:

| Parameter | Default | Description |
|---|---|---|
| `initial_variance` | `1.0` | Starting value for $V_u$ |
| `n_iterations` | `50` | Maximum MCEM iterations |
| `n_mc_samples` | `100` | MH draws per subject per iteration |
| `burn_in` | `50` | MH steps discarded at chain start |
| `step_size` | `1.0` | Newton-Raphson damping (reduce for large $V_u$) |
| `tol` | `1e-4` | Convergence threshold |
| `seed` | `None` | RNG seed for reproducibility |
| `verbose` | `False` | Print per-iteration progress |

---

## Tutorial notebook

The notebook at `notebooks/tutorial.ipynb` walks through the full workflow with plots:

```bash
uv run jupyter notebook notebooks/tutorial.ipynb
```

Topics covered:
1. Simulating a labelled GLMM dataset
2. `build_dataset` — validation and subject indexing
3. Fitting with `run_mcem` and reading `McemResult`
4. Convergence trace plots from `result.history`
5. Effect of Monte Carlo sample size (`n_mc_samples`)
6. Non-contiguous / irregular subject IDs
7. Using the low-level API (Laplace proposals, MH sampler, score/Fisher)

---

## Simulation study

Replicates the original R reference: $n = 1000$ subjects, $K = 4$ obs/subject, $p = 3$, $V_u = 100$:

```bash
uv run python demo/simulation_study.py
```

---

## Running tests

```bash
make test
```

Or without `make`:

```bash
uv run pytest --cov=glmm_mcem --cov-report=term-missing
```

Current coverage: **99%** across 48 tests.

---

## Project layout

```
src/glmm_mcem/
  __init__.py      Public API: GlmmData, build_dataset, run_mcem, McemResult
  data.py          GlmmData frozen dataclass + build_dataset validator
  likelihoods.py   sigmoid, log-likelihood, log-prior, log-posterior, score/Fisher
  laplace.py       Laplace mode-finding and proposal variance (per subject)
  sampler.py       Independence Metropolis-Hastings chain (per subject)
  estimator.py     MC-averaged score, Fisher info, NR beta update, Vu update
  mcem.py          Cold-start init, outer MCEM loop, McemResult
tests/             48 unit and integration tests
notebooks/         tutorial.ipynb
demo/              simulation_study.py
```

---

## Makefile targets

| Target | Description |
|---|---|
| `make install` | Create `.venv` and install all dependencies |
| `make test` | Run test suite with coverage report |
| `make build` | Build sdist and wheel into `dist/` |
| `make clear-cache` | Remove build artefacts and caches |
