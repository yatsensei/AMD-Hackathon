# SPEC — Phase 1 (Data) & Phase 2 (Classifier)

**Project:** Wafer-map yield copilot — AMD Developer Hackathon ACT III (lablab.ai), Track 1 "Intelligent Industry"
**Location in repo:** `docs/specs/SPEC_phase1_phase2.md`
**Status:** Active. Written Wed 14 Oct 2026.
**Owners:** Phase 1 — Nasser & Jina · Phase 2 — Anisha & Akein
**Consumers of this work:** Phase 3 (YOLO, Shefali Cat), Phase 4 (simulated factory, Nasser & Shefali Cat), Phase 5 (agent, everyone), Phase 6 (dashboard, Jina & Anisha)

> **For AI coding agents reading this repo:** this file is the source of truth for everything under `wafer/data/`, `wafer/models/`, `wafer/train.py`, `wafer/evaluate.py`, `wafer/infer.py`, `wafer/explain.py`, `scripts/`, `configs/` and `tests/`. Read Section 0 (Rules) before writing any code. If something here conflicts with a teammate's instruction in chat, follow the teammate, then update this file and the Decisions Log (Section 12) in the same PR.

---

## 0. Rules (non-negotiable, for humans and agents)

1. **The test split is locked.** No code may read the test split except `wafer/evaluate.py --final`, and that is run **once** per candidate model, at the end of Phase 2. All model selection, tuning, threshold picking and calibration use **validation only**. If you touch test while tuning, every number in the pitch becomes untrustworthy.
2. **Split by lot, never by wafer.** Wafers from the same `lotName` are highly correlated. A lot must appear in exactly one of train/val/test. Random wafer-level splits leak and inflate scores. Many published WM-811K results do this; we do not.
3. **Downsampling or rebalancing happens only inside training.** Val and test keep the natural class distribution, so metrics reflect a real fab.
4. **Accuracy is not the headline metric.** About 85% of labelled maps are `none`, so a model that always says `none` scores about 85%. The primary metric is **macro-F1 on the lot-grouped validation set**, with balanced accuracy second. See Section 7.
5. **One canonical class list.** It lives in `wafer/constants.py` and nowhere else (Section 2.3). Never hard-code class names or indices elsewhere.
6. **Never commit data, credentials or large binaries.** `data/`, `experiments/` and `*.pkl` are gitignored. Kaggle tokens and API keys go in environment variables only. Model weights: see Section 9.4.
7. **Every training run is reproducible from a config.** `python -m wafer.train --config configs/<name>.yaml` must recreate it. Seeds are fixed and logged.
8. **Stay inside your phase's folders.** If you need to change a shared interface (Section 9), say so in the team chat, update Section 9 and add a Decisions Log entry in the same PR.
9. **Save GPU credits.** Do preprocessing on any machine with CPU. Use the AMD GPU only for training and evaluation. Stop instances when idle.

---

## 1. Goal and success criteria

### 1.1 What we are building in Phases 1–2
A wafer-map defect-pattern classifier that:
- takes one wafer bin map (a 2D array of 0 = no die, 1 = pass, 2 = fail)
- returns one of 9 classes with a **calibrated** confidence
- **flags when it is unsure** (abstention / `needs_review`), so the agent in Phase 5 never sounds confident when the model isn't
- returns a **heatmap and bounding box** showing where on the wafer the evidence is (Grad-CAM). This replaces YOLO detection, which can't be trained because WM-811K has no boxes.
- runs on **CPU** fast enough for the dashboard (< 50 ms per wafer), and on the AMD GPU for training

### 1.2 What "winning" looks like for this model
WM-811K is a well-known benchmark, so a high number alone won't stand out. Our edge is a model that is **accurate and trustworthy under honest evaluation**:
- **Accurate:** strong macro-F1 on a lot-grouped split, with confidence intervals.
- **Calibrated:** when it says 90%, it is right about 90% of the time (low ECE).
- **Knows when it doesn't know:** a coverage–risk curve, e.g. "on the 92% of wafers it's confident about, it's 99% accurate; the rest go to an engineer."
- **Explainable:** a heatmap and box for every prediction.
- **Rigorously compared:** several architectures, losses and input encodings, multiple seeds, and statistical tests between them.

Pitch line we are aiming to be able to say truthfully:
> "Evaluated on a lot-grouped split so no lot leaks between train and test, our ensemble reaches **X** macro-F1 (95% CI **a–b**), is calibrated to **Y**% ECE, and routes the **Z**% of wafers it's unsure about to an engineer."

### 1.3 Targets (for steering effort, not promises)
| Metric (validation, lot-grouped) | Minimum to hand off | Stretch |
|---|---|---|
| Macro-F1 (9 classes) | ≥ 0.80 | ≥ 0.90 |
| Defect recall (any pattern vs `none`) | ≥ 0.95 | ≥ 0.98 |
| ECE after temperature scaling | ≤ 0.05 | ≤ 0.02 |
| Selective accuracy at 90% coverage | ≥ 0.97 | ≥ 0.99 |
| CPU inference per wafer | < 50 ms | < 10 ms |

Published single-model results on WM-811K are around 93% average F1, but usually with different (often non-grouped) splits. **Do not claim to beat published numbers.** Ours are not directly comparable, and we say so honestly.

---

## 2. Data facts every teammate and agent must know

### 2.1 Source
- Kaggle dataset `qingyi/wm811k-wafer-map`, file `LSWMD.pkl` (about 2 GB unpacked).
- 811,457 wafer maps from about 46,393 lots. **172,950 are labelled.** The rest are unlabelled (see stretch experiment E9).

### 2.2 Columns in `LSWMD.pkl`
| Column | Meaning | Gotchas |
|---|---|---|
| `waferMap` | 2D `uint8` array; 0 = outside wafer / no die, 1 = pass, 2 = fail | Sizes vary a lot (from very small to about 300 per side) |
| `dieSize` | Number of dies on the wafer | Float; useful as an auxiliary feature |
| `lotName` | Lot ID, e.g. `lot1` | **Used for the grouped split** |
| `waferIndex` | Wafer position in the lot (1–25) | Float |
| `trianTestLabel` | Original "Training"/"Test" tag | **Column name is misspelt in the dataset** (`trian`). Nested array; empty if unlabelled |
| `failureType` | Pattern label | **Nested array**, e.g. `[['Edge-Ring']]`; empty `[]` if unlabelled. Note `none` is lowercase |

### 2.3 Canonical classes (`wafer/constants.py`)
```python
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NONE_IDX = CLASS_TO_IDX["none"]
```
Expected labelled counts (verify in Phase 1 and record the real numbers in `docs/results/data_report.md`):
Center ≈ 4,294 · Donut ≈ 555 · Edge-Loc ≈ 5,189 · Edge-Ring ≈ 9,680 · Loc ≈ 3,593 · Near-full ≈ 149 · Random ≈ 866 · Scratch ≈ 1,193 · none ≈ 147,431.
**Near-full (about 149) and Donut (about 555) are tiny.** Expect only around 20 Near-full wafers in test, so their per-class scores will have wide confidence intervals. This is why we bootstrap (Section 7.3).

### 2.4 Known data quality issues
- **Label noise:** some labels are wrong or ambiguous. The error gallery (Section 7.5) will surface them. Do not "fix" labels in test.
- **Confusable pairs:** Edge-Loc ↔ Scratch, Edge-Loc ↔ Loc, Loc ↔ Center, Donut ↔ Edge-Ring. Track these explicitly.
- **Possible duplicate maps:** check with hashes (Section 3.6) and remove cross-split duplicates.
- **Orientation:** a pattern's meaning does not depend on rotation or flip (Center is still Center), but it **does depend on position**. Edge vs centre matters, so **never use random crops or translations.**

---

## 3. Phase 1 — Data pipeline (Nasser & Jina)

**Deadline:** Thursday 15 Oct, 12:00 Sydney time. Phase 2 cannot start training until 3.8 is done.
**Definition of done:** every item in Section 3.10 is checked and `pytest tests/test_data.py` passes.

### 3.1 Environment
- Python 3.11. Phase 1 needs CPU only: `numpy pandas scikit-learn opencv-python-headless pyarrow tqdm pyyaml pillow matplotlib`.
- Pin versions in `requirements.txt`. The GPU/ROCm PyTorch install is separate (Section 4.1).

### 3.2 Download — `scripts/download_data.sh`
```bash
# Needs KAGGLE_USERNAME and KAGGLE_KEY in env (never commit them)
mkdir -p data/raw
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw --unzip
ls -lh data/raw   # expect LSWMD.pkl
```
Record the file's SHA-256 in `docs/results/data_report.md` so everyone can confirm they have the same file.

### 3.3 Load with the old-pandas compatibility patch — `wafer/data/load.py`
The pickle was written with an old pandas version and fails to load on modern pandas without a patch.
```python
import sys
import numpy as np
import pandas as pd

def load_raw(path="data/raw/LSWMD.pkl") -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError:
        import pandas.core.indexes as _idx
        sys.modules["pandas.indexes"] = _idx          # old module path used by the pickle
        # if a further ModuleNotFoundError names another old path, alias it the same way
        return pd.read_pickle(path)

def _flat(x):
    """Turn nested arrays like [['Edge-Ring']] into 'Edge-Ring'; empty -> None."""
    a = np.asarray(x, dtype=object).ravel()
    return str(a[0]) if a.size else None
```
Add one row to the Decisions Log noting the exact patch that worked and your pandas version.

### 3.4 Clean — `wafer/data/preprocess.py::clean()`
1. `df["label"] = df["failureType"].map(_flat)`; `df["orig_split"] = df["trianTestLabel"].map(_flat)`.
2. Keep labelled rows only (`label` is not None) and assert `set(df.label) == set(CLASSES)`.
3. Unlabelled rows go to a separate `data/interim/unlabeled_meta.parquet` (metadata only, for stretch E9).
4. Cast `waferIndex` to int and `dieSize` to float.
5. Add `uid` = row index of the original dataframe (stable ID used everywhere).
6. Add `orig_h`, `orig_w` = shape of `waferMap`.
7. Fill missing `lotName` with `f"nolot_{uid}"` (treated as a singleton lot). Log how many were filled.
8. Save the labelled originals (unresized) to `data/interim/labeled.pkl`, so maps can be re-resized at other resolutions without reloading the raw file.

### 3.5 Resize — `wafer/data/preprocess.py::resize_map()`
Two encodings are produced. Phase 2 will compare them (experiment E1).

**Encoding A — "nearest" (the team's original plan):** `cv2.resize(map, (S, S), interpolation=cv2.INTER_NEAREST)`. Values stay exactly in {0,1,2}. Then one-hot to 3 channels (no-die, pass, fail).

**Encoding B — "area-fraction" (recommended):** nearest-neighbour downsampling can **delete thin scratches**, because a 1-die-wide line can fall between sampled pixels. Instead build two masks and resize each:
- `wafer_mask = (map > 0)`, `fail_mask = (map == 2)` as float32
- if the original is larger than S, use `cv2.INTER_AREA` (each output pixel = fraction of dies in that block); otherwise use `cv2.INTER_NEAREST`
- channels = `[wafer_mask, fail_mask, fail_mask * wafer_mask_inverse_safe]` — in practice store `[wafer_frac, fail_frac]` and derive `pass_frac = wafer_frac - fail_frac` in the dataset class, giving 3 channels in [0,1]

Default `S = 64`. Also produce `S = 96` for experiment E4, only if time allows.

Unit-test both: shape (S, S); encoding A values ⊆ {0,1,2}; encoding B values in [0,1]; and a synthetic 1-die-wide diagonal scratch on a 200×200 wafer must survive encoding B (fail_frac > 0 along the line) — this test documents why B exists.

### 3.6 Deduplicate — `wafer/data/preprocess.py::dedupe()`
- Hash each **original** map: `hashlib.sha1(map.tobytes() + str(map.shape).encode())`.
- Report how many exact duplicates exist, and how many have conflicting labels.
- After splitting (3.7), **drop from val and test any map whose hash also appears in train.** Keep them in train. Log counts in the data report.

### 3.7 Split — `wafer/data/split.py`
Lot-grouped **and** stratified, 70/15/15:
```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
fold = np.empty(len(df), dtype=int)
for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["lotName"])):
    fold[idx] = k
df["split"] = np.select([fold < 3, fold < 6], ["test", "val"], default="train")  # 3/20 test, 3/20 val
```
Then **assert**:
- no `lotName` appears in more than one split
- every class appears in every split (Near-full included); if not, change `random_state` and log it
- each split's class proportions are within ±2 percentage points of the overall proportions (Near-full and Donut excepted; log their counts)

Also save `cv_fold` (0–13) for train-only lots, using a second `StratifiedGroupKFold(n_splits=5)` on the train rows, for optional cross-validation (Section 6.4).

**Do not downsample `none` here.** That's done per epoch in training (Section 5.3).

### 3.8 Save outputs
| Path | Contents | Consumer |
|---|---|---|
| `data/processed/wafers_s64.npz` | `X_nearest` uint8 (N,64,64), `X_area` uint8 (N,2,64,64) scaled ×255, `y` int64, `uid` int64 — **all labelled wafers**, aligned by row. Use `np.savez_compressed` | Phase 2 |
| `data/processed/metadata.parquet` | `uid, lotName, waferIndex, label, label_idx, split, cv_fold, orig_h, orig_w, dieSize, hash, orig_split` | Phases 2, 3, 4 |
| `yolo_data/{train,val,test}/<class>/<uid>.png` | 64×64 PNGs rendered with `wafer/viz.py::render_wafer()` (same colours as the dashboard) | Phase 3 |
| `docs/results/data_report.md` | File hash, class counts per split, duplicates removed, lots filled, maps-per-size histogram | Pitch + everyone |

**For Phase 3, the YOLO `test/` folder exists but Shefali must only score on it once, with the same rule as Section 0.1.** YOLO's `val` should point to our `val`.

### 3.9 Shared rendering — `wafer/viz.py`
```python
PALETTE = {0: (20, 20, 20), 1: (60, 160, 90), 2: (220, 60, 60)}  # no-die, pass, fail
def render_wafer(map2d: np.ndarray, size: int | None = None) -> np.ndarray: ...  # returns HxWx3 uint8
```
The YOLO PNGs, the dashboard (Phase 6) and pitch figures must all use this function, so wafers look the same everywhere.

### 3.10 Phase 1 checklist
- [ ] `scripts/download_data.sh` works from a clean machine (with Kaggle env vars)
- [ ] `wafer/data/load.py` loads the pickle; the patch is documented
- [ ] Labelled count = 172,950 (or the real number is logged with an explanation)
- [ ] Both encodings produced; scratch-survival test passes
- [ ] Dedup done; counts logged
- [ ] Lot-grouped split; all assertions pass
- [ ] `wafers_s64.npz`, `metadata.parquet`, `yolo_data/` written
- [ ] `docs/results/data_report.md` written
- [ ] `pytest tests/test_data.py` passes
- [ ] Phase 2 and Phase 4 owners told the data is ready (post in team chat + tick in `docs/STATUS.md`)

---

## 4. Phase 2 — Environment on AMD (Anisha & Akein)

### 4.1 PyTorch on ROCm (MI300X on AMD Developer Cloud)
- Use the ROCm PyTorch build or AMD's ROCm PyTorch Docker image on the instance. Don't pip-install the default CUDA wheel.
- Sanity check (put this in `scripts/check_gpu.py`):
```python
import torch
print(torch.__version__, "hip:", torch.version.hip)        # hip should be non-None on ROCm
print("gpu available:", torch.cuda.is_available())           # ROCm exposes the GPU through the torch.cuda API
print(torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda"); print((x @ x).sum().item())
```
- In code, always use `device = "cuda" if torch.cuda.is_available() else "cpu"`. On ROCm, `"cuda"` means the AMD GPU. **Never** call CUDA-only libraries (e.g. NVIDIA-specific kernels or `nvidia-smi` in scripts). Use `rocm-smi` to monitor the GPU.
- Use `torch.autocast("cuda", dtype=torch.bfloat16)` for mixed precision. If bf16 causes problems, fall back to fp32 and log it.
- Extra packages for Phase 2: `torch torchvision timm scikit-learn matplotlib pyyaml pandas tqdm` (+ `statsmodels` for McNemar).
- Record in the Decisions Log: ROCm version, PyTorch version, GPU name. The pitch needs "trained on AMD Instinct MI300X with ROCm" and it must be true.

### 4.2 Data loading on the GPU box
- The whole 64×64 dataset fits in RAM. Load `wafers_s64.npz` once into tensors; avoid per-sample disk reads.
- `num_workers=4–8`, `pin_memory=True`, `persistent_workers=True`. If DataLoader workers misbehave, set `num_workers=0`; the data is small enough.
- Do augmentation on the GPU in batches (Section 5.4) — it's faster than per-sample CPU augmentation.

---

## 5. Phase 2 — Model and training

### 5.1 Code layout
```
wafer/
  constants.py        # CLASSES (Section 2.3)
  viz.py              # render_wafer (Section 3.9)
  data/
    load.py  preprocess.py  split.py
    dataset.py        # WaferDataset: encoding A or B -> float tensor (3,S,S)
    augment.py        # D4 group, small rotation, die-flip noise (GPU, batched)
    sampler.py        # per-epoch 'none' downsampling
  models/
    small_cnn.py      # team baseline
    resnet.py         # ResNet-18 with a small-image stem
    factory.py        # build_model(cfg)
  train.py            # python -m wafer.train --config configs/e0_baseline.yaml
  evaluate.py         # val by default; --final for test (Section 0.1)
  calibrate.py        # temperature scaling on val
  metrics.py          # every metric in Section 7
  explain.py          # Grad-CAM + bbox
  infer.py            # classify(), explain() public API (Section 9)
configs/              # one YAML per experiment
scripts/              # download_data.sh, prepare_data.py, check_gpu.py, run_experiments.sh
tests/                # test_data.py, test_model.py, test_infer.py, test_metrics.py
experiments/          # gitignored; one folder per run
docs/results/         # committed: reports, figures, leaderboard.csv, MODEL_CARD.md
```

### 5.2 Input
`WaferDataset(encoding="nearest"|"area", split=..., size=64)` returns `(x, y, uid)`:
- encoding A: one-hot of {0,1,2} → 3 channels {no-die, pass, fail}
- encoding B: `[wafer_frac, pass_frac, fail_frac]` from the stored uint8 /255
- Optional auxiliary scalars (experiment E8): `log(dieSize)`, `orig_h/orig_w`, fail fraction. Concatenated after global pooling.

### 5.3 Class imbalance (train only)
Two strategies; compare in E2.
- **Per-epoch `none` downsampling** (`sampler.py`): each epoch uses all defect wafers plus a fresh random sample of 15,000 `none` wafers. Unlike a fixed downsample, this sees all `none` data over training.
- **Class-weighted loss**: weights `w_c = (N / n_c) ** 0.5`, normalised to mean 1. The square root stops Near-full from dominating.
Never rebalance val or test.

### 5.4 Augmentation (train only, applied on GPU per batch)
- **D4 symmetries:** random choice of the 8 rotations by 90° × flips. Exact on a square grid, so no interpolation artefacts.
- **Small rotation:** ±15°, nearest-neighbour, then re-apply the wafer mask so no dies appear outside the wafer. Probability 0.5. Ablate in E3b.
- **Die-flip noise:** inside the wafer, flip pass↔fail on 0–0.5% of dies at random. Builds robustness to random test noise; must not turn `none` into `Random`, so keep the rate very low and ablate.
- **Forbidden:** random crops, translations, scaling, colour jitter. Position is part of the label (Center vs Edge-Loc).

### 5.5 Architectures
| ID | Model | Notes |
|---|---|---|
| `small_cnn` | 4 blocks of [Conv3×3–BN–ReLU ×2 → MaxPool2], channels 32→64→128→256, global average pool, dropout 0.3, FC→9 | The team baseline. About 1.2M params |
| `resnet18s` | torchvision `resnet18(weights=None)`, replace `conv1` with 3×3 stride 1 and set `maxpool = nn.Identity()` | Standard "small image" ResNet. Keeps 64×64 detail that the ImageNet stem would throw away |
| `resnet18s_pt` | Same, but initialised from ImageNet weights (copy conv1 weights averaged to the new 3×3 where possible, or re-init conv1 only) | Tests whether pretraining helps on this very non-natural image domain |
| `effnet_b0` / `convnext_atto` (timm) | `timm.create_model(name, in_chans=3, num_classes=9)` | Only if time allows (E3) |

Every model exposes `forward(x) -> logits` and `features(x)` (last conv map, needed for Grad-CAM).

### 5.6 Training loop (`wafer/train.py`)
- Optimiser AdamW, lr 1e-3, weight decay 1e-4; cosine schedule with 1 epoch linear warm-up.
- Batch size 256; up to 40 epochs; **early stopping on val macro-F1**, patience 7; keep the best checkpoint.
- Loss options: `ce`, `weighted_ce`, `focal` (γ = 2, with class weights as α), `ce_ls` (label smoothing 0.05).
- bf16 autocast; gradient clipping at 1.0.
- Seeds: set Python, NumPy and Torch seeds; `torch.backends.cudnn.benchmark = True` is fine (determinism not required across GPUs, but log the seed).
- Each run writes `experiments/<run_id>/` with: `config.yaml`, `best.pt`, `history.csv` (per-epoch train loss, val loss, val macro-F1, val balanced accuracy, lr), `val_metrics.json`, `val_probs.npy` + `val_uids.npy` (needed for calibration, ensembling and McNemar), and `env.json` (GPU, ROCm, torch versions, git commit).
- At the end, append one row to `docs/results/leaderboard.csv`: `run_id, experiment, model, encoding, loss, sampler, aug, seed, val_macro_f1, val_bal_acc, val_defect_recall, val_ece_raw, epochs, train_minutes, git_commit`.

### 5.7 Config example — `configs/e0_baseline.yaml`
```yaml
experiment: E0
model: small_cnn
encoding: nearest
size: 64
loss: weighted_ce
sampler: none_downsample   # or: none
none_per_epoch: 15000
aug: {d4: true, small_rot: false, die_flip: 0.0}
optim: {lr: 1.0e-3, wd: 1.0e-4, epochs: 40, batch_size: 256, warmup_epochs: 1, patience: 7}
seeds: [0, 1, 2]
aux_features: false
```
`scripts/run_experiments.sh` runs every config in a list, one after another, so it can be left running overnight.

---

## 6. Phase 2 — Experiment plan

Work in tiers. **Don't start a tier until the one before it is done and handed off.**

### 6.1 Tier 0 — must have (Thursday 15 Oct, by 22:00)
| ID | What | Why |
|---|---|---|
| E0 | `small_cnn`, encoding A, weighted CE, D4 aug, 3 seeds | Baseline. **Hand this off to Phases 5 and 6 immediately** (Section 9) so nobody waits on us |

### 6.2 Tier 1 — the core comparison (Friday 16 Oct)
Each row is 3 seeds. Change **one thing at a time** from the current best, so differences are attributable.
| ID | Variable | Options |
|---|---|---|
| E1 | Input encoding | nearest (A) vs area-fraction (B) |
| E2 | Imbalance + loss | weighted CE · focal · CE + label smoothing with `none` downsampling · weighted CE with `none` downsampling |
| E3 | Architecture | `small_cnn` vs `resnet18s` vs `resnet18s_pt` |
| E3b | Augmentation | D4 only vs D4 + small rotation vs D4 + rotation + die-flip |
| E5 | Test-time augmentation (TTA) | Average softmax over the 8 D4 transforms (no retraining needed) |
| E6 | Ensemble | Average calibrated probabilities of the best 3–5 runs (different seeds and/or architectures) |
| — | Calibration | Temperature scaling on every finalist (Section 7.4) |

**Selection rule:** the winner of each row is the option with the highest **mean val macro-F1 across 3 seeds**. If two options are within one standard deviation, prefer the simpler/faster one. Log the choice in the Decisions Log.

**Model freeze: Friday 16 Oct, 21:00.** Whatever is best at that point is exported as v1 for the agent and dashboard (Section 9.4). Tier 2 work may only replace it if it beats v1 on val **and** passes all tests before Saturday 10:00.

### 6.3 Tier 2 — stretch (only if Tier 1 is frozen and handed off)
| ID | What | Notes |
|---|---|---|
| E4 | Resolution 96 vs 64 | Needs `wafers_s96.npz` from Phase 1 |
| E7 | Other backbones (`effnet_b0`, `convnext_atto`) | Via timm |
| E8 | Auxiliary scalar features (die size, aspect ratio, fail fraction) | Concatenated after pooling |
| E9 | Use the ~638k **unlabelled** maps | (a) Self-supervised pretraining (SimCLR-style with D4 + die-flip views) then fine-tune, or (b) pseudo-labelling with high-confidence predictions. **Exclude every unlabelled wafer whose `lotName` is in val or test** — otherwise information from val/test lots leaks into training |
| E10 | Mixed-pattern stress test with MixedWM38 | Not for training. Run the final model on MixedWM38 maps and check it gives **low confidence / needs_review** on mixed wafers instead of confidently picking one pattern. Report the abstention rate. This is a strong "knows its limits" story |

### 6.4 Optional cross-validation
If GPU time allows, run the final configuration with 5-fold lot-grouped CV on train+val (`cv_fold`) and report mean ± SD macro-F1. This is the most robust estimate of performance and is good for the slides. The test split is still never used for this.

---

## 7. Phase 2 — Evaluation (`wafer/metrics.py`, `wafer/evaluate.py`)

### 7.1 Metrics computed for every run (val) and for the final model (test, once)
**Headline**
- Macro-F1 (primary) · balanced accuracy · defect recall (binary: any pattern vs `none`) · defect precision

**Full picture**
- Per-class precision, recall, F1, support
- Overall accuracy and weighted F1 (report, but never headline)
- Matthews correlation coefficient (MCC) and Cohen's kappa
- Top-2 accuracy
- One-vs-rest ROC-AUC and PR-AUC (macro); PR-AUC is the more informative one under imbalance
- Confusion matrix as counts **and** row-normalised (recall view)
- Confusable-pair error rates: Edge-Loc↔Scratch, Edge-Loc↔Loc, Loc↔Center, Donut↔Edge-Ring

**Trustworthiness**
- Negative log-likelihood and multi-class Brier score
- Expected calibration error (ECE, 15 bins) before and after temperature scaling, plus a reliability diagram
- Coverage–risk curve: accuracy and macro-F1 on the most-confident X% of wafers, for X from 50% to 100%
- Selective accuracy at 90% and 95% coverage

### 7.2 Slices (robustness tests)
Report macro-F1 by:
- original wafer size bins (`orig_h * orig_w` in quartiles) — small wafers lose detail when resized
- `dieSize` quartiles
- `waferIndex` (early vs late in lot) — sanity check, should be flat
- D4 consistency: fraction of test wafers whose prediction is the same under all 8 symmetries (should be very high; TTA helps)
- Noise robustness: macro-F1 when 0.5% / 1% / 2% of dies are flipped
- Sanity inputs: an all-pass wafer → `none`; an all-fail wafer → `Near-full`; an empty map → error is handled gracefully

### 7.3 Statistics
- **Bootstrap CIs by lot:** resample lots with replacement 1,000 times; report 95% CI for macro-F1, balanced accuracy and per-class F1. Resampling lots (not wafers) respects the correlation inside a lot.
- **McNemar test** between the final model and the E0 baseline on the same wafers. This is what lets us say the improvement is statistically significant.
- Across seeds: report mean ± SD for every Tier 1 row.

### 7.4 Calibration and abstention — `wafer/calibrate.py`
1. Fit a single temperature `T` on val logits by minimising NLL (LBFGS). For an ensemble, fit `T` on the averaged logits.
2. Choose the abstention threshold `τ` on val: the smallest calibrated max-probability such that selective accuracy on val ≥ 0.98 (log the coverage this gives).
3. `needs_review = (max_prob < τ) or (top1 and top2 are a confusable pair and top1 − top2 < 0.15)`.
4. Save `T`, `τ` and the margin rule into `models/<version>/calibration.json`.

### 7.5 Figures and reports (`docs/results/<run_or_final>/`)
- `confusion_counts.png`, `confusion_norm.png`
- `reliability_before.png`, `reliability_after.png`
- `coverage_risk.png`
- `per_class.csv`, `metrics.json`, `slices.csv`
- `error_gallery.png`: the 30 most confident **wrong** predictions, rendered with `render_wafer`, labelled "true / predicted / confidence". Review these by eye; many will be label noise, which is a useful finding for the pitch.
- `gradcam_examples.png`: one correct example per class with heatmap and box

### 7.6 The final test evaluation (once)
```bash
python -m wafer.evaluate --final --model models/v1   # refuses to run without --final
```
- Writes `docs/results/final/` with everything in 7.1–7.5 plus bootstrap CIs and the McNemar result.
- Also evaluates on the dataset's original `trianTestLabel == "Test"` subset **for reference only**, clearly labelled as non-grouped and not comparable.
- Generates `docs/results/MODEL_CARD.md` (Section 10).
- **Only numbers in `docs/results/final/` may go in the pitch.**

---

## 8. Phase 2 — Explainability (`wafer/explain.py`)

YOLO detection can't be trained on WM-811K because it has no bounding boxes. We get localisation another way:
1. **Grad-CAM** on the last conv layer for the predicted class → a 64×64 heatmap, upsampled to the original wafer size.
2. **Bounding box:** threshold the heatmap at 0.5 of its max, intersect with fail dies (`map == 2`), take the largest connected component (`cv2.connectedComponentsWithStats`), and return its box in **original wafer coordinates**. For `none`, return no box. For `Random` and `Near-full`, return the wafer's bounding box, since those patterns cover the whole wafer.
3. For an ensemble or TTA, compute Grad-CAM on the single best member (it's for explanation, not scoring).

This gives the dashboard the "box around the defect" the team wanted, without needing box labels.

---

## 9. Interfaces for other phases (contract — don't change without Section 0.8)

### 9.1 `wafer/infer.py`
```python
from wafer.infer import load_model, classify, explain

model = load_model("models/v1")          # loads weights + calibration.json; CPU by default, "cuda" if available

classify(wafer_map: np.ndarray) -> dict
# wafer_map: raw 2D array of 0/1/2, any size (resizing happens inside)
# returns:
{
  "pattern": "Edge-Ring",                # top class (always present; Phase 5/6 rely on this key)
  "confidence": 0.94,                    # calibrated probability of "pattern"
  "probs": {"Center": 0.01, ...},        # all 9 calibrated probabilities, sum to 1
  "top2": [["Edge-Ring", 0.94], ["Donut", 0.03]],
  "needs_review": False,                 # Section 7.4 rule; the agent must say it's unsure when True
  "model_version": "v1"
}

explain(wafer_map: np.ndarray) -> dict
{
  "heatmap": np.ndarray,                 # float32, same H×W as input, values in [0,1]
  "bbox": [x0, y0, x1, y1] | None,       # original coordinates; None for "none"
  "pattern": "Edge-Ring"
}

classify_batch(maps: list[np.ndarray]) -> list[dict]   # same output per item, for the stream replay
```
- Must run on **CPU** in < 50 ms per wafer (tested). The dashboard machine may not have a GPU.
- Must never raise on odd input sizes; on an invalid map (wrong dtype, values outside {0,1,2}, empty), raise `ValueError` with a clear message.
- Phase 5's `classify_wafer` tool wraps `classify()` directly. Phase 6 uses `classify()`, `explain()` and `wafer.viz.render_wafer()`.

### 9.2 Mock for early integration
On Thursday, before any model exists, provide `wafer/infer_mock.py` with the same signatures, returning plausible fixed outputs. Phases 5 and 6 build against it and swap the import later.

### 9.3 Data contract for Phase 4
`data/processed/metadata.parquet` (Section 3.8). Phase 4 builds `stream.csv` **only from `split == "test"` wafers**, so the demo shows the model on wafers it has never seen. Phase 4 may group them into fictional lots; keep `uid` so each streamed wafer can be traced back.

### 9.4 Model release
`models/v1/` contains `weights.pt` (or several for an ensemble), `config.yaml`, `calibration.json`, `classes.json`, `metrics_val.json`, `README.md`.
- If the folder is < 50 MB, commit it. Otherwise upload it as a GitHub Release asset and add `scripts/fetch_model.sh` that downloads it.
- Bump to `v2` only through the freeze rule in 6.2.

### 9.5 Benchmarking YOLO fairly (for Phase 3)
`wafer/evaluate.py --predictions path/to/preds.csv` must accept a CSV of `uid, prob_<class>...` so YOLO results are scored with **exactly the same metrics and splits** as the CNN. Then "we benchmarked two model families" is an honest comparison.

---

## 10. Model card — `docs/results/MODEL_CARD.md` (generated, then edited by hand)
Sections: intended use (assist yield engineers; not an autonomous decision-maker) · training data and split method · architecture and training details · hardware (AMD Instinct MI300X, ROCm version) · final test metrics with 95% CIs · calibration and abstention behaviour · known limitations (label noise; single-label only; trained on one fab's data from WM-811K; small classes have wide CIs; mixed-pattern wafers are out of scope and should be flagged) · how to reproduce.

---

## 11. Tests (`tests/`, run with `pytest -q`; CI optional)
**`test_data.py`**
- loader returns a dataframe with the expected columns; labelled count matches the data report
- `_flat` handles `[['x']]`, `['x']`, `[]`
- resize: shapes, value sets, scratch-survival (3.5)
- split: no lot overlap; every class in every split; proportions within tolerance
- no cross-split duplicate hashes in val/test
- npz rows align with metadata `uid`s

**`test_model.py`**
- every model in the factory maps `(B,3,64,64)` → `(B,9)` and exposes `features()`
- one training step on a tiny batch reduces loss (smoke test)
- augmentation keeps the wafer mask and the label unchanged; D4 transforms are exact (applying all 8 and inverting returns the original)

**`test_metrics.py`**
- metrics on a hand-made example match hand-computed values
- ECE is 0 for a perfectly calibrated toy case
- bootstrap-by-lot returns intervals containing the point estimate

**`test_infer.py`**
- `classify` returns all keys with correct types; `probs` sums to 1 ± 1e-5
- works on maps of size 10×10, 64×64, 300×200
- invalid input raises `ValueError`
- CPU latency < 50 ms per wafer (average over 100)
- sanity inputs from 7.2
- `explain` returns a heatmap of the input's shape and a box inside the wafer

**Rule for agents:** don't mark a checklist item done if its tests fail. Never weaken or delete a test to make it pass; fix the code or raise it in chat.

---

## 12. Decisions log
Append one line per decision: `date · who · decision · reason`. Agents must add entries for any deviation from this spec.

| Date | Who | Decision | Reason |
|---|---|---|---|
| 14 Oct | spec | Lot-grouped stratified split instead of wafer-level stratified | Wafer-level splits leak correlated wafers from the same lot and inflate scores |
| 14 Oct | spec | Rebalance only inside training; val/test keep natural distribution | Metrics must reflect a real fab |
| 14 Oct | spec | Macro-F1 primary; accuracy reported but never headline | 85% of labels are `none` |
| 14 Oct | spec | Add area-fraction encoding alongside nearest | Nearest-neighbour downsampling can erase thin scratches |
| 14 Oct | spec | Grad-CAM + box instead of YOLO detection | WM-811K has no box labels |
| 14 Oct | spec | No random crops/translations in augmentation | Defect position is part of the label |
| | | | |

---

## 13. Status board — `docs/STATUS.md`
Keep a short checklist in `docs/STATUS.md` (one line per item from 3.10, 6.1–6.3, 7.6, 9.1–9.4) with owner and ✅/🟡/❌. Update it in every PR. It's how teammates and their agents see what's done without asking.

## 14. Timeline (Sydney time)
| When | Milestone |
|---|---|
| Wed 14 Oct night | Repo scaffold, `constants.py`, `viz.py`, GPU check passes, data downloaded |
| Thu 15 Oct 12:00 | Phase 1 complete (3.10) |
| Thu 15 Oct 14:00 | `infer_mock.py` available to Phases 5/6 |
| Thu 15 Oct 22:00 | E0 baseline trained, evaluated on val, exported as `v0` |
| Fri 16 Oct 21:00 | Tier 1 done; best model calibrated and frozen as `v1` |
| Sat 17 Oct 10:00 | Final test evaluation run once; `docs/results/final/` + model card committed |
| Sat 17 Oct 19:00 | Team submission deadline (team plan). Official deadline: Mon 19 Oct 02:00 Sydney |

## 15. Pitch-ready claims checklist
Before any number goes on a slide, confirm:
- [ ] it comes from `docs/results/final/` (test, run once), or is explicitly labelled as validation/CV
- [ ] it has a 95% CI (bootstrap by lot) where relevant
- [ ] the slide says the split is lot-grouped
- [ ] any comparison to published numbers says they used different splits and are not directly comparable
- [ ] the AMD hardware/software claim matches `env.json`
