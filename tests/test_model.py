"""Spec 11 test_model.py: models, augmentation, sampler/prior, input encodings."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wafer.constants import CLASSES, NONE_IDX
from wafer.data import augment, sampler
from wafer.data.dataset import base_from_area_u8, base_from_nearest, radius_channel, to_input
from wafer.losses import make_loss
from wafer.models.factory import ENCODING_CHANNELS, build_model

MODELS = ["small_cnn", "resnet18s", "resnet18s1"]


def _circle_base(b=4, s=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
    wafer = ((yy - s / 2 + 0.5) ** 2 + (xx - s / 2 + 0.5) ** 2 <= (s / 2 - 2) ** 2).float()
    wafer = wafer.expand(b, s, s).clone()
    fail = (torch.rand(b, s, s, generator=g) < 0.1).float() * wafer
    return torch.stack([wafer, fail], 1)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("encoding", ["area", "nearest", "area_r"])
def test_models_map_to_logits_and_expose_16px_cam_layer(name, encoding):
    net = build_model({"model": name, "encoding": encoding}).eval()
    seen = {}
    net.cam_layer.register_forward_hook(lambda m, i, o: seen.__setitem__("shape", o.shape))
    out = net(torch.rand(2, ENCODING_CHANNELS[encoding], 64, 64))
    assert out.shape == (2, len(CLASSES))
    assert seen["shape"][-1] >= 16 and seen["shape"][-2] >= 16
    names = list(net.state_dict())
    assert not any(k.startswith("cam_layer") for k in names), "cam_layer must not duplicate weights"


def test_aux_features_change_head_input():
    net = build_model({"model": "small_cnn", "aux_features": True}).eval()
    assert net(torch.rand(2, 3, 64, 64), torch.rand(2, 3)).shape == (2, 9)


@pytest.mark.parametrize("loss", ["ce", "weighted_ce", "focal", "ce_ls"])
def test_one_training_step_reduces_loss(loss):
    torch.manual_seed(0)
    net = build_model({"model": "small_cnn"})
    crit = make_loss(loss, torch.ones(9))
    x = to_input(_circle_base(16), "area")
    y = torch.randint(0, 9, (16,))
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    net.train()
    losses = []
    for _ in range(5):
        l_ = crit(net(x), y)
        opt.zero_grad()
        l_.backward()
        opt.step()
        losses.append(l_.item())
    assert losses[-1] < losses[0]


def test_d4_is_exact_and_invertible():
    x = torch.rand(3, 2, 8, 8)
    outs = [augment.apply_d4(x, t) for t in range(8)]
    assert all(torch.equal(augment.invert_d4(o, t), x) for t, o in enumerate(outs))
    assert len({tuple(o.flatten().tolist()) for o in outs}) == 8     # 8 distinct transforms


@pytest.mark.parametrize("cfg", [{"d4": True}, {"d4": True, "small_rot": True},
                                 {"d4": True, "small_rot": True, "die_flip": 0.005}])
def test_augment_keeps_fail_inside_wafer(cfg):
    torch.manual_seed(1)
    base = _circle_base(32)
    out = augment.augment(base, cfg)
    assert out.shape == base.shape
    assert (out[:, 1] <= out[:, 0] + 1e-6).all()                     # pass >= 0
    assert (out[:, 1][out[:, 0] == 0] == 0).all()                    # no fails outside the wafer
    if not cfg.get("small_rot"):
        # D4 and die-flip keep the wafer mask's size exactly
        assert torch.equal(out[:, 0].sum((1, 2)), base[:, 0].sum((1, 2)))


def test_small_rotation_moves_mask_but_keeps_its_size_roughly():
    torch.manual_seed(0)
    base = _circle_base(8)
    out = augment.small_rotation(base, max_deg=15, p=1.0)
    rel = (out[:, 0].sum((1, 2)) / base[:, 0].sum((1, 2)) - 1).abs()
    assert (rel < 0.03).all()


def test_die_flip_rate_is_low_and_only_inside_wafer():
    torch.manual_seed(0)
    base = _circle_base(64)
    out = augment.die_flip(base, 0.005)
    changed = (out[:, 1] != base[:, 1])
    assert (changed & (base[:, 0] == 0)).sum() == 0
    assert changed.float().sum() / base[:, 0].sum() < 0.005


def test_encodings_channels():
    u8 = torch.tensor([[[[255, 0]], [[128, 0]]]], dtype=torch.uint8)       # (1,2,1,2)
    base = base_from_area_u8(u8)
    a = to_input(base, "area")
    assert torch.allclose(a[:, 0], a[:, 1] + a[:, 2])
    near = base_from_nearest(torch.tensor([[[0, 1, 2]]]))
    n = to_input(near, "nearest")
    assert torch.equal(n.sum(1), torch.ones(1, 1, 3))                       # one-hot
    assert to_input(_circle_base(2), "area_r").shape[1] == 4


def test_radius_channel_is_d4_equivariant():
    base = _circle_base(2)
    r = radius_channel(base[:, 0])
    for t in range(8):
        assert torch.allclose(radius_channel(augment.apply_d4(base, t)[:, 0]), augment.apply_d4(r, t), atol=1e-4)
    centre, edge = r[0, 32, 32], r[0, 32, 3]
    assert centre < 0.1 and 0.9 < edge < 1.1


def test_sampler_prior_matches_hand_computation():
    counts = np.array([10, 0, 0, 0, 0, 0, 0, 0, 90], float)
    w = sampler.class_weights(counts, "weighted_ce")
    assert np.isclose(w.mean(), 1.0)
    prior = sampler.effective_prior(counts, "none_downsample", 30, "weighted_ce")
    assert np.isclose(prior.sum(), 1.0)
    raw = np.array([10, 0, 0, 0, 0, 0, 0, 0, 30]) * w
    assert np.allclose(prior, raw / raw.sum())
    assert np.allclose(sampler.effective_prior(counts, "none", 30, "ce"), counts / 100)


def test_epoch_indices_downsample_none_and_keep_all_defects():
    y = np.array([0] * 5 + [NONE_IDX] * 100)
    idx = np.arange(105)
    out = sampler.epoch_indices(idx, y, "none_downsample", 20, np.random.default_rng(0))
    assert len(out) == 25 and set(range(5)) <= set(out) and len(set(out)) == 25
