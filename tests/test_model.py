import numpy as np
import torch

from mvtree.constants import NUM_CLASSES, SPECIES
from mvtree.losses import LogitAdjustedCrossEntropy, class_prior_from_labels
from mvtree.model import MultiViewTreeClassifier


def test_forward_shapes():
    model = MultiViewTreeClassifier(backbone="convnextv2_atto", pretrained=False).eval()
    B, S = 2, 64
    with torch.no_grad():
        out = model(torch.randn(B, 4, 3, S, S), torch.randn(B, 2, 3, S, S), torch.randn(B, 3, S, S),
                    torch.rand(B, 1) * 30)
    assert out.shape == (B, NUM_CLASSES)


def test_parameter_names_match_release():
    names = {n.split(".")[0] for n, _ in MultiViewTreeClassifier(backbone="convnextv2_atto", pretrained=False)
             .named_parameters()}
    assert names == {"branch_side", "branch_topbot", "branch_detail", "classifier"}


def test_logit_adjustment():
    logits, y = torch.randn(16, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (16,))
    uniform = LogitAdjustedCrossEntropy(np.ones(NUM_CLASSES))
    assert torch.allclose(uniform(logits, y), torch.nn.functional.cross_entropy(logits, y), atol=1e-6)
    prior = class_prior_from_labels([0, 0, 0, 1], NUM_CLASSES)
    assert abs(prior.sum() - 1) < 1e-9 and prior[0] > prior[1] > prior[2]


def test_species_order_is_sorted():
    assert SPECIES == sorted(SPECIES) and len(SPECIES) == 33
