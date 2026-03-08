"""Tests for glmm_mcem.data — validation contract of build_dataset."""

import numpy as np
import pytest
from glmm_mcem import build_dataset


def test_rejects_2d_y():
    with pytest.raises(ValueError, match="1-D"):
        build_dataset(np.zeros((5, 2)), np.zeros((5, 2)), np.zeros(5))


def test_rejects_1d_X():
    with pytest.raises(ValueError, match="2-D"):
        build_dataset(np.zeros(5), np.zeros(5), np.zeros(5))


def test_rejects_row_mismatch():
    with pytest.raises(ValueError, match="rows"):
        build_dataset(np.zeros(5), np.zeros((3, 2)), np.zeros(5))


def test_rejects_bad_subject_ids_shape():
    with pytest.raises(ValueError, match="shape"):
        build_dataset(np.zeros(5), np.zeros((5, 2)), np.zeros((5, 1)))


def test_rejects_non_binary_y():
    with pytest.raises(ValueError, match="0 and 1"):
        build_dataset(np.array([0.0, 1.0, 2.0]), np.zeros((3, 2)), np.zeros(3))
