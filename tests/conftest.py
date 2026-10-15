import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def cloud():
    """A small synthetic tree: a vertical trunk plus a noisy crown, with exact duplicates."""
    rng = np.random.default_rng(0)
    trunk = np.c_[rng.normal(0, 0.1, (300, 2)), rng.uniform(0, 4, 300)]
    crown = np.c_[rng.normal(0, 1.5, (1500, 2)), rng.uniform(3, 9, 1500)]
    pts = np.vstack([trunk, crown, crown[:50]]).astype(np.float32)  # duplicates exercise depth ties
    return pts.astype(np.float64)
