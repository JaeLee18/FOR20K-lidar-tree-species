"""Three-stream multi-view classifier (ConvNeXtV2-Base x 3).

Seven renders are split into three view groups, each with its own backbone:

    side   (4 views) -> backbone_side   -> max-pool over views -> 1024-d
    topbot (2 views) -> backbone_topbot -> max-pool over views -> 1024-d
    detail (1 view)  -> backbone_detail                        -> 1024-d

The three group features are concatenated with the scalar tree height (m)
and classified by LayerNorm -> Dropout -> Linear(3073, 512) -> GELU ->
Dropout -> Linear(512, 33). Total: 264.7M parameters.

Parameter names match the released checkpoints, so do not rename attributes.
"""

from __future__ import annotations

from typing import List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DEFAULT_BACKBONE, NUM_CLASSES


class MultiViewTreeClassifier(nn.Module):
    """Args:
        num_classes: Number of species (33 for FOR-species20K).
        backbone: Any timm model name. The paper uses ConvNeXtV2-Base; the
            DenseNet-201 and DINOv2-Base ablations use ``densenet201`` and
            ``vit_base_patch14_dinov2.lvd142m`` (with ``img_size=224``).
        dropout: Dropout in the classification head.
        pretrained: Load timm's pretrained weights (needed for training only).
        img_size: Resize inputs to this size before the backbone. Only needed
            for ViT backbones with a fixed token grid; ``None`` for CNNs.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, backbone: str = DEFAULT_BACKBONE,
                 dropout: float = 0.3, pretrained: bool = True,
                 img_size: Optional[int] = None) -> None:
        super().__init__()
        extra = {} if img_size is None else {"img_size": img_size}
        self.img_size = img_size
        self.branch_side = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **extra)
        self.branch_topbot = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **extra)
        self.branch_detail = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **extra)
        self.feat_dim = self.branch_side.num_features

        combined_dim = self.feat_dim * 3 + 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _extract(self, branch: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.img_size is not None and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return branch(x)

    def forward(self, side_views: torch.Tensor, topbot_views: torch.Tensor,
                detail_view: torch.Tensor, tree_height: torch.Tensor) -> torch.Tensor:
        """Args:
            side_views: (B, 4, 3, H, W)
            topbot_views: (B, 2, 3, H, W)
            detail_view: (B, 3, H, W)
            tree_height: (B, 1), metres
        Returns:
            (B, num_classes) logits (unadjusted; apply softmax for probabilities).
        """
        B = side_views.shape[0]
        side = self._extract(self.branch_side, side_views.flatten(0, 1)).view(B, 4, -1)
        topbot = self._extract(self.branch_topbot, topbot_views.flatten(0, 1)).view(B, 2, -1)
        detail = self._extract(self.branch_detail, detail_view)
        features = torch.cat([side.max(dim=1).values, topbot.max(dim=1).values, detail, tree_height], dim=1)
        return self.classifier(features)

    def head_parameters(self) -> List[nn.Parameter]:
        return list(self.classifier.parameters())

    def backbone_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.startswith("classifier")]
