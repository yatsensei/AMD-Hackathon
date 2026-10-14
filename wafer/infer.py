"""Public inference API (spec 9.1).

    from wafer.infer import load_model
    clf = load_model("models/v1", device="cpu")
    clf.classify(wafer_map) / clf.classify_batch(maps) / clf.explain(wafer_map)

    from wafer.infer import classify, classify_batch, explain   # default model, loaded lazily:
    # $WAFER_MODEL_DIR, else models/v1, else models/v0

Preprocessing calls the same functions as the data pipeline, and calibration/decision code is
shared with evaluation, so what the agent sees is exactly what was scored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from wafer import calibrate
from wafer.constants import CLASS_TO_IDX
from wafer.data import preprocess
from wafer.data.dataset import aux_features, base_from_area_u8, base_from_nearest, to_input
from wafer.decision import DEFAULT_MARGIN, result_dict
from wafer.env import REPO_ROOT
from wafer.explain import defect_bbox, grad_cam, to_original
from wafer.models.factory import build_model
from wafer.predict import forward_base


def load_members(model_dir: Path, cfg: dict, device: str) -> list[torch.nn.Module]:
    from safetensors.torch import load_file
    members = []
    for i, mcfg in enumerate(cfg["members"]):
        net = build_model(mcfg)
        net.load_state_dict(load_file(str(model_dir / "weights" / f"m{i}.safetensors")))
        members.append(net.to(device).eval())
    return members


class WaferClassifier:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.dir = Path(model_dir)
        self.cfg = yaml.safe_load((self.dir / "config.yaml").read_text())
        self.cal = json.loads((self.dir / "calibration.json").read_text())
        self.device = device
        self.encoding, self.size = self.cfg["encoding"], int(self.cfg["size"])
        self.tta, self.aux = bool(self.cfg.get("tta", False)), bool(self.cfg.get("aux_features", False))
        self.model_version = self.cfg["version"]
        self.members = load_members(self.dir, self.cfg, device)

    # -- preprocessing: identical to the npz path (uint8 storage included)
    def _base(self, maps: list[np.ndarray]) -> torch.Tensor:
        if self.encoding == "nearest":
            x = np.stack([preprocess.encode_nearest(m, self.size) for m in maps])
            return base_from_nearest(torch.from_numpy(x).to(self.device))
        x = np.stack([preprocess.area_to_uint8(preprocess.encode_area(m, self.size)) for m in maps])
        return base_from_area_u8(torch.from_numpy(x).to(self.device))

    def _aux(self, maps):
        if not self.aux:
            return None
        a = aux_features([(m > 0).sum() for m in maps], [m.shape[0] for m in maps],
                         [m.shape[1] for m in maps], [(m == 2).sum() for m in maps])
        return torch.from_numpy(a).to(self.device)

    def outputs(self, maps: list[np.ndarray]) -> np.ndarray:
        """Combined, *uncalibrated* outputs z (spec 7.4 step 1) for validated maps."""
        base, aux = self._base(maps), self._aux(maps)
        outs = [forward_base(m, base, self.encoding, aux, tta=self.tta).cpu().numpy() for m in self.members]
        return calibrate.combine_members(outs, self.tta)

    def probs(self, maps: list[np.ndarray]) -> np.ndarray:
        return calibrate.apply(self.cal, self.outputs(maps))

    def _result(self, p: np.ndarray) -> dict:
        return result_dict(p, self.cal["theta_none"], self.cal["tau"], self.model_version,
                           self.cal.get("margin", DEFAULT_MARGIN))

    def classify(self, wafer_map: np.ndarray) -> dict:
        return self._result(self.probs([preprocess.validate_map(wafer_map)])[0])

    def classify_batch(self, maps: list[np.ndarray], batch_size: int = 512) -> list[dict]:
        valid = [preprocess.validate_map(m) for m in maps]
        out = []
        for i in range(0, len(valid), batch_size):
            out += [self._result(p) for p in self.probs(valid[i:i + batch_size])]
        return out

    def explain(self, wafer_map: np.ndarray) -> dict:
        m = preprocess.validate_map(wafer_map)
        pattern = self.classify(m)["pattern"]
        x = to_input(self._base([m]), self.encoding)
        net = self.members[0]                        # spec 8.3: explain with one member
        aux = self._aux([m])
        model = net if aux is None else _WithAux(net, aux)
        cam = grad_cam(model, x, CLASS_TO_IDX[pattern])
        heat = to_original(cam, m.shape) * (m > 0)
        return {"heatmap": heat.astype(np.float32), "bbox": defect_bbox(m, heat, pattern), "pattern": pattern}


class _WithAux(torch.nn.Module):
    def __init__(self, net, aux):
        super().__init__()
        self.net, self.aux = net, aux

    @property
    def cam_layer(self):
        return self.net.cam_layer

    def forward(self, x):
        return self.net(x, self.aux)


def load_model(path: str | Path = "models/v1", device: str = "cpu") -> WaferClassifier:
    p = Path(path)
    return WaferClassifier(p if p.is_absolute() or p.exists() else REPO_ROOT / p, device)


_default: WaferClassifier | None = None


def default_model_dir() -> Path:
    if os.environ.get("WAFER_MODEL_DIR"):
        return Path(os.environ["WAFER_MODEL_DIR"])
    for v in ("v1", "v0"):
        if (REPO_ROOT / "models" / v / "config.yaml").exists():
            return REPO_ROOT / "models" / v
    raise FileNotFoundError("no model found: set WAFER_MODEL_DIR or export models/v0 (python -m wafer.export)")


def _model() -> WaferClassifier:
    global _default
    if _default is None:
        _default = WaferClassifier(default_model_dir(), "cpu")
    return _default


def classify(wafer_map: np.ndarray) -> dict:
    return _model().classify(wafer_map)


def classify_batch(maps: list[np.ndarray]) -> list[dict]:
    return _model().classify_batch(maps)


def explain(wafer_map: np.ndarray) -> dict:
    return _model().explain(wafer_map)
