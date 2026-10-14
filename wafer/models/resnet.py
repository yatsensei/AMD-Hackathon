"""ResNet-18 with a small-image stem (spec 5.5).

stride=2 (`resnet18s`, default): 0.56 GMACs at 64x64; layer2 is 16x16 -> cam_layer.
stride=1 (`resnet18s1`): 2.2 GMACs; layer3 is 16x16 -> cam_layer. Serve without TTA only.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ResNet18S(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 9, stem_stride: int = 2, n_aux: int = 0):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_chans, 64, 3, stride=stem_stride, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.layer1, m.layer2, m.layer3, m.layer4)
        self.fc = nn.Linear(512 + n_aux, num_classes)
        self._cam_index = 4 if stem_stride == 2 else 5        # body[4] = layer2, body[5] = layer3

    @property
    def cam_layer(self) -> nn.Module:
        """The 16x16 stage. A property so its weights are not registered twice."""
        return self.body[self._cam_index]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        z = self.features(x).mean(dim=(2, 3))
        if aux is not None:
            z = torch.cat([z, aux], dim=1)
        return self.fc(z)
