"""Logit-adjusted cross-entropy (Menon et al., ICLR 2021, arXiv:2007.07314)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedCrossEntropy(nn.Module):
    """Cross-entropy on ``logits + tau * log(prior)``.

    Rare classes get an effective margin without resampling or class weights.
    The prior should come from the current fold's training labels only.
    Predict with the raw, unadjusted logits.
    """

    def __init__(self, class_prior, tau: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        prior = torch.as_tensor(np.asarray(class_prior), dtype=torch.float32)
        if prior.ndim != 1 or (prior < 0).any():
            raise ValueError("class_prior must be a non-negative 1-D vector")
        prior = prior / prior.sum().clamp_min(1e-12)
        self.register_buffer("log_prior", torch.log(prior.clamp_min(1e-12)))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.tau * self.log_prior, target,
                               label_smoothing=self.label_smoothing)


def class_prior_from_labels(labels, num_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1e-12)
    return counts / counts.sum()
