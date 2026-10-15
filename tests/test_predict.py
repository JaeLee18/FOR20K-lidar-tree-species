"""End-to-end prediction on the bundled examples (skipped when weights are absent)."""
import os

import numpy as np
import pytest

from mvtree.weights import find_models

pytestmark = pytest.mark.skipif(not find_models(), reason="model weights not installed")


@pytest.fixture(scope="module")
def predictor():
    from mvtree.predict import Predictor
    return Predictor(folds=[0], tta=2)


@pytest.mark.parametrize("name", ["05554_Pinus_sylvestris_ULS.laz", "03450_Quercus_robur_MLS.laz"])
def test_examples(root, predictor, name):
    res = predictor.predict_file(os.path.join(root, "examples", name))
    assert res.species == "_".join(name.split("_")[1:3])
    assert abs(res.probs.sum() - 1) < 1e-4


def test_deterministic(root, predictor):
    path = os.path.join(root, "examples", "05554_Pinus_sylvestris_ULS.laz")
    a, b = predictor.predict_file(path).probs, predictor.predict_file(path).probs
    assert np.allclose(a, b, atol=1e-5)
