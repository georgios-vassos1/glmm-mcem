"""Data structures for the GLMM-MCEM package.

A logistic GLMM with a single random intercept per subject has the form:

    Y_ij | u_i  ~ Bernoulli( sigmoid(X_ij @ beta + u_i) )
    u_i          ~ N(0, Vu)

This module defines the :class:`GlmmData` container and the
:func:`build_dataset` factory that validates and indexes raw arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class GlmmData:
    """Immutable container for a grouped binary-response dataset.

    Parameters
    ----------
    y : ndarray of shape (n_obs,)
        Binary response vector (0 / 1).
    X : ndarray of shape (n_obs, p)
        Design matrix (fixed effects only; **no** random-effect column).
    subject_ids : ndarray of shape (n_obs,) of int
        Integer subject identifier for each observation (0-indexed).
    subject_index : dict mapping subject_id -> list[int]
        Pre-computed grouping: ``subject_index[i]`` is the list of row
        indices that belong to subject *i*.
    n_subjects : int
        Number of distinct subjects.
    n_obs : int
        Total number of observations.
    p : int
        Number of fixed-effect predictors (columns of *X*).
    """

    y: np.ndarray
    X: np.ndarray
    subject_ids: np.ndarray
    subject_index: Dict[int, List[int]]
    n_subjects: int
    n_obs: int
    p: int


def build_dataset(y: np.ndarray, X: np.ndarray, subject_ids: np.ndarray) -> GlmmData:
    """Validate inputs and construct a :class:`GlmmData`.

    Parameters
    ----------
    y : array-like of shape (n_obs,)
        Binary response (values must be 0 or 1).
    X : array-like of shape (n_obs, p)
        Fixed-effects design matrix.  Must **not** contain a random-effect
        column — the random intercept is handled internally.
    subject_ids : array-like of shape (n_obs,)
        Subject identifier for each observation.  Will be re-mapped to
        contiguous integers starting from 0.

    Returns
    -------
    GlmmData

    Raises
    ------
    ValueError
        If shapes are inconsistent or *y* contains values outside {0, 1}.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    subject_ids = np.asarray(subject_ids)

    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    n_obs = len(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] != n_obs:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {n_obs} elements"
        )

    if subject_ids.shape != (n_obs,):
        raise ValueError(
            f"subject_ids must have shape ({n_obs},), got {subject_ids.shape}"
        )

    unique_vals = np.unique(y)
    if not set(unique_vals.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"y must contain only 0 and 1; found {unique_vals}")

    # Re-map subject IDs to contiguous 0-based integers.
    unique_subjects = np.unique(subject_ids)
    id_map = {old: new for new, old in enumerate(unique_subjects)}
    remapped = np.array([id_map[s] for s in subject_ids], dtype=int)

    n_subjects = len(unique_subjects)
    subject_index: Dict[int, List[int]] = {i: [] for i in range(n_subjects)}
    for row, subj in enumerate(remapped):
        subject_index[subj].append(row)

    return GlmmData(
        y=y,
        X=X,
        subject_ids=remapped,
        subject_index=subject_index,
        n_subjects=n_subjects,
        n_obs=n_obs,
        p=X.shape[1],
    )
