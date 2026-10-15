"""Team baseline (spec 5.5): 4 blocks of [Conv3x3-BN-ReLU] x2, max-pool after blocks 1-3 only
(so the final map is 8x8, not 4x4, at the same FLOPs), GAP, dropout 0.3, FC -> classes."""
from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 9, widths=(32, 64, 128, 256),
                 dropout: float = 0.3, n_aux: int = 0, cam_block: int = 4):
        super().__init__()
        c = [in_chans, *widths]
        self.blocks = nn.ModuleList(_block(c[i], c[i + 1]) for i in range(4))
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(widths[-1] + n_aux, num_classes))
        self._cam_block = cam_block

    @property
    def cam_layer(self) -> nn.Module:
        """Block `cam_block` (1-based), before its pool: 64, 32, 16, 8 px for blocks 1..4.
        Default 4 (8x8, the most class-specific): the heatmap only has to *choose* the fail-die
        cluster, whose own extent sets the box (spec 8 / Decisions Log). A property, not an
        attribute, so the weights are not registered twice."""
        return self.blocks[self._cam_block - 1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i < 3:
                x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        z = self.features(x).mean(dim=(2, 3))
        if aux is not None:
            z = torch.cat([z, aux], dim=1)
        return self.head(z)
