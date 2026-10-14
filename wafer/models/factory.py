from __future__ import annotations

from torch import nn

from wafer.constants import CLASSES

ENCODING_CHANNELS = {"area": 3, "nearest": 3, "area_r": 4}


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]
    in_chans = ENCODING_CHANNELS[cfg.get("encoding", "area")]
    n_aux = 3 if cfg.get("aux_features") else 0
    if name == "small_cnn":
        from wafer.models.small_cnn import SmallCNN
        return SmallCNN(in_chans=in_chans, num_classes=len(CLASSES), n_aux=n_aux)
    if name in ("resnet18s", "resnet18s1"):
        from wafer.models.resnet import ResNet18S
        return ResNet18S(in_chans=in_chans, num_classes=len(CLASSES), n_aux=n_aux,
                         stem_stride=2 if name == "resnet18s" else 1)
    raise ValueError(f"unknown model {name!r}; options: small_cnn, resnet18s, resnet18s1")
