"""glmm_mcem — MCEM for logistic GLMMs with a normal random intercept.

Public API
----------
GlmmData
    Immutable dataset container.
build_dataset
    Validate and index raw arrays into a :class:`GlmmData`.
run_mcem
    Fit the model by Monte Carlo EM.
McemResult
    Result container returned by :func:`run_mcem`.
"""

from .data import GlmmData, build_dataset
from .mcem import McemResult, run_mcem

__all__ = ["GlmmData", "build_dataset", "run_mcem", "McemResult"]
