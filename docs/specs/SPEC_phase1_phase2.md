# SPEC — Phase 1 (Data) & Phase 2 (Classifier)

**Project:** Wafer-map yield copilot — AMD Developer Hackathon ACT III (lablab.ai), Track 1 "Intelligent Industry"
**Location in repo:** `docs/specs/SPEC_phase1_phase2.md`
**Status:** Active — **v2**, revised Wed 14 Oct 2026 after design review. The reasoning for every change from v1 is in `docs/specs/REVIEW_phase1_phase2.md`.
**Owners:** Phase 1 — Nasser & Jina · Phase 2 — Anisha & Akein
**Consumers of this work:** Phase 3 (YOLO, Shefali Cat), Phase 4 (simulated factory, Nasser & Shefali Cat), Phase 5 (agent, everyone), Phase 6 (dashboard, Jina & Anisha)

> **For AI coding agents reading this repo:** this file is the source of truth for everything under `wafer/data/`, `wafer/models/`, `wafer/train.py`, `wafer/evaluate.py`, `wafer/infer.py`, `wafer/explain.py`, `scripts/`, `configs/` and `tests/`. Read Section 0 (Rules) before writing any code. If something here conflicts with a teammate's instruction in chat, follow the teammate, then update this file and the Decisions Log (Section 12) in the same PR.

---

## 0. Rules (non-negotiable, for humans and agents)

1. **The test split is locked.** Only `wafer/evaluate.py --final` may read it, and it runs **once**, on **one** frozen model (v1, or v2 via the freeze rule in 6.2). The comparators named in Section 7.6 (E0 baseline, E0b classical baseline, Phase 3 YOLO) are scored in that same run so we can compare against them. **They are never candidates to choose between on test.** All model selection, tuning, threshold picking and calibration use **validation only**.
2. **Split by lot, never by wafer.** Wafers from the same `lotName` are highly correlated. A lot must appear in exactly one of train/val/test. Random wafer-level splits leak and inflate scores. Many published WM-811K results do this; we do not, and experiment E-LEAK measures how much it matters.
3. **Downsampling or rebalancing happens only inside training.** Val and test keep the natural class distribution. Rebalancing shifts the class priors the model assumes, so it **must be undone at inference** (Section 7.4).
4. **Accuracy is not the headline metric.** About 85% of labelled maps are `none`, so a model that always says `none` scores about 85%. The primary metric is **macro-F1 (9 classes) on the lot-grouped validation set**, always reported next to **defect macro-F1 (the 8 defect classes)** and defect recall. See Section 7.
5. **One canonical class list.** It lives in `wafer/constants.py` and nowhere else (Section 2.3). Never hard-code class names or indices elsewhere.
6. **Never commit data, credentials or large binaries.** `data/`, `experiments/` and `*.pkl` are gitignored. Kaggle tokens and API keys go in environment variables only. Model weights: see Section 9.4. **Check the SHA-256 of `LSWMD.pkl` before unpickling it**, because pickles can run code when loaded. Load our own weights with `safetensors` or `torch.load(..., weights_only=True)`.
7. **Every training run is reproducible from a config.** `python -m wafer.train --config configs/<name>.yaml` must recreate it. Seeds are fixed and logged. Runs from a dirty git tree are marked as such in `env.json`.
8. **Stay inside your phase's folders.** If you need to change a shared interface (Section 9), say so in the team chat, update Section 9 and add a Decisions Log entry in the same PR.
9. **Use the GPU for what needs a GPU.** Do preprocessing on any CPU machine. On the MI300X, run many seeds or configs **in parallel** rather than one after another. Stop instances when idle.
10. **Report what you ship.** The configuration evaluated on test (members × TTA × preprocessing) is exactly the one behind `wafer.infer`, and it must meet the CPU latency budget (Section 9.1). A config that's too slow to serve isn't eligible for v1, however good its numbers.

---

## 1. Goal and success criteria

### 1.1 What we are building in Phases 1–2
A wafer-map defect-pattern classifier that:
- takes one wafer bin map (a 2D array of 0 = no die, 1 = pass, 2 = fail)
- returns one of 9 classes with a **calibrated** probability. Calibrated means for the *natural* class mix of a real fab, not the rebalanced training mix.
- uses an explicit **operating point** for "defect vs no defect", set to the fab's priority: missing a defect costs more than a false alarm
- **flags when it is unsure** (`needs_review`, plus the reasons), so the agent in Phase 5 never sounds confident when the model isn't
- returns a **heatmap and box** showing which fail dies drove the decision (Grad-CAM attribution). This replaces YOLO detection, which can't be trained because WM-811K has no boxes. Its quality is measured on a synthetic-insertion benchmark (Section 8).
- runs on **CPU** within the latency budget (Section 9.1) for the dashboard, and trains on the AMD GPU

### 1.2 What "winning" looks like for this model
WM-811K is a well-known benchmark, so a high number alone won't stand out. Our edge is a model that is **accurate and trustworthy under honest evaluation**:
- **Accurate:** strong macro-F1 on a lot-grouped split, with lot-bootstrap confidence intervals.
- **Honest about leakage:** E-LEAK shows how much a wafer-random split would have inflated our own score.
- **Calibrated:** when it says 90%, it is right about 90% of the time, **including on defect wafers**, not just the easy `none` majority.
- **Knows when it doesn't know:** a coverage–risk curve on predicted-defect wafers, plus a per-class review rate.
- **Explainable:** a heatmap and box for every prediction, with localisation measured rather than just shown.
- **Rigorously compared:** baselines (including a classical one), 5 seeds, and paired lot-bootstrap tests.

Pitch line we are aiming to be able to say truthfully:
> "Evaluated on a lot-grouped split (we checked: a wafer-random split gives the same score, so there's no leakage hiding in the number), our model reaches **X** macro-F1 (95% CI **a–b**), catches **R**% of defective wafers, is calibrated to **Y**% ECE on defect predictions, and routes the **Z**% of wafers it's unsure about to an engineer."

### 1.3 Targets (for steering effort, not promises)
| Metric (validation, lot-grouped) | Minimum to hand off | Stretch |
|---|---|---|
| Macro-F1 (9 classes) | ≥ 0.80 | ≥ 0.90 |
| Defect macro-F1 (8 defect classes) | ≥ 0.75 | ≥ 0.88 |
| Defect recall at the operating point (any pattern vs `none`) | ≥ 0.95 | ≥ 0.98 |
| Defect precision at that operating point | reported | ≥ 0.90 |
| ECE on predicted-defect wafers, after calibration | ≤ 0.05 | ≤ 0.03 |
| Selective accuracy on predicted-defect wafers at 90% defect coverage | ≥ 0.95 | ≥ 0.98 |
| CPU latency, served config (Section 9.1 reference setup) | < 50 ms | < 10 ms |

Overall ECE and overall selective accuracy are still reported. They aren't targets, because the `none` majority makes them easy to pass.
Published single-model results on WM-811K are around 93% average F1, but usually with different (often non-grouped) splits. **Do not claim to beat published numbers.** Ours aren't directly comparable, and E-LEAK shows why.

---

## 2. Data facts every teammate and agent must know

### 2.1 Source
- Kaggle dataset `qingyi/wm811k-wafer-map`, file `LSWMD.pkl` (about 2 GB unpacked).
- 811,457 wafer maps from about 46,393 lots. **172,950 are labelled.** The rest are unlabelled (see E9).

### 2.2 Columns in `LSWMD.pkl`
| Column | Meaning | Gotchas |
|---|---|---|
| `waferMap` | 2D `uint8` array; 0 = outside wafer / no die, 1 = pass, 2 = fail | Sizes vary a lot (from very small to about 300 per side) |
| `dieSize` | Number of dies on the wafer | Float. Also a stand-in for *product*; see the shortcut warning in E8 |
| `lotName` | Lot ID, e.g. `lot1` | **Used for the grouped split** |
| `waferIndex` | Wafer position in the lot (1–25) | Float |
| `trianTestLabel` | Original "Training"/"Test" tag | **Column name is misspelt in the dataset** (`trian`). Nested array; empty if unlabelled. **Not used for evaluation** (see 7.6) |
| `failureType` | Pattern label | **Nested array**, e.g. `[['Edge-Ring']]`; empty `[]` if unlabelled. Note `none` is lowercase |

### 2.3 Canonical classes (`wafer/constants.py`)
```python
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NONE_IDX = CLASS_TO_IDX["none"]
DEFECT_IDX = [i for i, c in enumerate(CLASSES) if c != "none"]
CONFUSABLE_PAIRS = [("Edge-Loc", "Scratch"), ("Edge-Loc", "Loc"),
                    ("Loc", "Center"), ("Donut", "Edge-Ring")]
```
Expected labelled counts (verify in Phase 1 and record the real numbers in `docs/results/data_report.md`):
Center ≈ 4,294 · Donut ≈ 555 · Edge-Loc ≈ 5,189 · Edge-Ring ≈ 9,680 · Loc ≈ 3,593 · Near-full ≈ 149 · Random ≈ 866 · Scratch ≈ 1,193 · none ≈ 147,431 (25,519 defects in total).
**Near-full (about 149) and Donut (about 555) are tiny.** What matters for their confidence intervals is **how many distinct lots** they come from, not how many wafers. Phase 1 must report lots per class (3.7).

### 2.4 Known data quality issues
- **Label noise:** some labels are wrong or ambiguous. Duplicates with conflicting labels (3.6) give a measured lower bound, and the error gallery (7.5) shows examples. Do not "fix" labels in test.
- **Confusable pairs:** see `CONFUSABLE_PAIRS`. Track these explicitly.
- **Duplicate maps:** check with hashes (3.6).
- **Orientation:** a pattern's meaning doesn't depend on rotation or flip (Center is still Center), but it **does depend on position**. Edge vs centre matters, so **never use random crops or translations.**
- **Product shortcut:** lots are grouped, but products (wafer geometry) are shared across splits. A model can learn "this product tends to get pattern X". E11 measures this.

---

## 3. Phase 1 — Data pipeline (Nasser & Jina)

**Critical-path deadline:** Thursday 15 Oct, **10:00** Sydney time: `wafers_s64.npz` + `metadata.parquet` (3.4–3.8). Phase 2 is blocked until then.
**Non-critical (Thu 15 Oct, 16:00):** `yolo_data/` PNGs, data-report figures, `wafers_s96.npz`.
**Definition of done:** every item in Section 3.10 is checked and `pytest tests/test_data.py` passes.

### 3.1 Environment
- CPU machine: Python ≥ 3.10 with `numpy pandas scikit-learn opencv-python-headless pyarrow tqdm pyyaml pillow matplotlib`. Versions in `requirements.txt` are bounded (so one file installs on Python 3.10–3.13); the exact versions that built the data are recorded in the data report.
- GPU box: use the Python that ships in the ROCm PyTorch image. **Don't try to force a specific Python version there.** Outputs are `.npz`/`.parquet`, which work across versions.

### 3.2 Download — `scripts/download_data.sh`
```bash
# Needs KAGGLE_API_TOKEN (or legacy KAGGLE_USERNAME + KAGGLE_KEY) in env; never commit them
mkdir -p data/raw
kaggle datasets download qingyi/wm811k-wafer-map -p data/raw --unzip   # kaggle>=2: no -d flag
sha256sum data/raw/LSWMD.pkl   # compare with docs/results/data_report.md before loading
```
The first person to download records the SHA-256 in `docs/results/data_report.md`. Everyone else checks against it **before** unpickling.

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
Add one row to the Decisions Log noting the exact patch that worked and your pandas version. After 3.4, nothing else loads the raw pickle.

### 3.4 Clean — `wafer/data/preprocess.py::clean()`
1. `df["label"] = df["failureType"].map(_flat)`; `df["orig_split"] = df["trianTestLabel"].map(_flat)`.
2. Keep labelled rows only (`label` is not None) and assert `set(df.label) == set(CLASSES)`.
3. Unlabelled rows go to `data/interim/unlabeled_meta.parquet` (uid, lotName, waferIndex, shape, dieSize). Their maps go to `data/interim/unlabeled_s64.npz` **only when E9 starts**.
4. Cast `waferIndex` to int and `dieSize` to float.
5. Add `uid` = row index of the original dataframe (stable ID used everywhere).
6. Add `orig_h`, `orig_w` = shape of `waferMap`; `n_die = (map > 0).sum()`; `n_fail = (map == 2).sum()`.
7. Fill missing `lotName` with `f"nolot_{uid}"` (treated as a singleton lot). Log how many were filled.
8. Save the labelled originals (unresized) to `data/interim/labeled.pkl`, so maps can be re-resized at other resolutions without reloading the raw file.

### 3.5 Resize and encode — `wafer/data/preprocess.py`
**Encoding B, "area-fraction" (default, from E0 onwards):**
```python
wafer = (m > 0).astype(np.float32)
fail  = (m == 2).astype(np.float32)
wafer_frac = cv2.resize(wafer, (S, S), interpolation=cv2.INTER_AREA)  # works both down- and up-sampling
fail_frac  = cv2.resize(fail,  (S, S), interpolation=cv2.INTER_AREA)
# store [wafer_frac, fail_frac] as uint8 (x255). The dataset derives 3 channels:
#   [wafer_frac, pass_frac = wafer_frac - fail_frac, fail_frac]   all in [0,1]
```
**Why B is the default:** nearest-neighbour downsampling can delete thin scratches. Measured (`docs/specs/evidence/bench_models.py`): a 1-die diagonal scratch on a 200×200 wafer keeps 45/140 fail pixels under nearest. Shifted one die sideways, it keeps **0**. Area-fraction keeps 126 pixels both times.

**Encoding A, "nearest" (comparator for E1 only):** `cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)`, values in {0,1,2}, one-hot to 3 channels.

**Radius channel (for E1c, computed in the dataset class, not stored):** `r = distance from the wafer_frac centroid / wafer radius`, clipped to [0, 1.2]. It doesn't change under D4.

Default `S = 64`. Also produce `S = 96` (non-critical) for E4.

Unit tests: shape (S, S); A ⊆ {0,1,2}; B in [0,1]; `pass_frac ≥ 0`; **scratch survival** for both the on-grid and the one-die-offset diagonal (B must keep fail_frac > 0 along the whole line; the test documents that A doesn't).

### 3.6 Deduplicate by linking lots — `wafer/data/preprocess.py::link_lots()`
- Hash each **original** map: `hashlib.sha1(map.tobytes() + str(map.shape).encode())` (as uint8).
- A map is **informative** if `min(n_fail, n_pass) ≥ 1` (`--dup-min-minority`). Identical *uniform* maps (all-pass, all-fail) are expected between different physical wafers of the same product and are ignored. An identical map with both pass and fail dies is treated as the same wafer recorded twice.
- **Before splitting**, lots that share an informative map are merged into one `split_group` (union-find). The split (3.7) groups by `split_group`, so every copy of a duplicated wafer lands in the same split. **Nothing is dropped**, and the seed search sees the final class counts. After splitting, assert that no informative map appears in two splits.
- Report: duplicate groups (all / informative), groups with **conflicting labels** (the measured label-noise floor), class mix, lots linked, and the largest linked group. The pipeline warns if one group holds > 1% of wafers. If that happens, consider `--dup-min-minority 2` and log it.

### 3.7 Split — `wafer/data/split.py`
Grouped by `split_group` (a lot, or lots linked by duplicate maps, 3.6) **and** stratified, 70/15/15, with an **automatic, logged seed search**:
```python
from sklearn.model_selection import StratifiedGroupKFold

def make_split(df, seed):
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["split_group"])):
        fold[idx] = k
    return np.select([fold < 3, fold < 6], ["test", "val"], default="train")  # 3/20, 3/20

# for seed in range(100): score = max over (class, split in {val,test}) of
#     |share_in_split - share_overall| / share_overall
# keep the lowest-score split that satisfies MIN_COUNT[class] in both val and test
# (default MIN_COUNT: 15 for Near-full, 50 for Donut, 100 otherwise; adjust after seeing lots-per-class)
```
The full search runs in parallel (`--jobs`). Measured on a real-scale synthetic set (173k labelled wafers): about 2.5 min on 4 cores, and the whole pipeline in under 4 min at 3.3 GB peak RAM for 373k raw rows. Expect roughly 6–8 GB for the real 811k rows. The chosen seed and parameters go to `data/processed/split_params.json`. Then **assert**:
- no `lotName` or `split_group` appears in more than one split; no informative map appears in two splits
- every class meets `MIN_COUNT` in val and test
- relative class-share deviation ≤ 25% per class per split (log the actual worst case)

**Report lots per class** in the data report (`df.groupby("label").lotName.nunique()`). A synthetic worst case (`docs/specs/evidence/split_sim.py`) put all Near-full in test and none in val. The seed search is there to stop that silently happening on real data.

`cv_fold` (0–4): a second `StratifiedGroupKFold(n_splits=5)` over **train + val** split groups, for optional CV (6.4) and E11. Test lots get `cv_fold = -1`.

**Do not downsample `none` here.** That's done per epoch in training (Section 5.3).

### 3.8 Save outputs
| Path | Contents | Consumer | Critical path? |
|---|---|---|---|
| `data/processed/wafers_s64.npz` | `X_area` uint8 (N,2,64,64) ×255, `X_nearest` uint8 (N,64,64), `y` int64, `uid` int64 — **every valid labelled wafer** (sorted by `uid`), aligned by row with `metadata.parquet`. `np.savez_compressed` | Phase 2 | **Yes** |
| `data/processed/metadata.parquet` | `uid, lotName, split_group, waferIndex, label, label_idx, split, cv_fold, orig_h, orig_w, n_die, n_fail, dieSize, hash, dup_group, orig_split` | Phases 2, 3, 4 | **Yes** |
| `data/processed/split_params.json` | chosen split seed, seeds tried / feasible, `min_count`, `max_rel_dev`, `cv_seed`, `dup_min_minority` | tests, reproducibility | **Yes** |
| `docs/results/data_report.md` | File SHA-256, class counts **and lot counts** per split, chosen split seed and its worst-case deviation, duplicates (total / conflicting / dropped), lots filled, map-size histogram | Pitch + everyone | Counts yes, figures no |
| `yolo_data/{train,val,test}/<class>/<uid>.png` | 64×64 PNGs from `wafer/viz.py::render_wafer()`. Phase 3 chooses whether to render from A or B (see review §3) | Phase 3 | No |

**Phase 3 scores on its `test/` folder only through the single final run (Rule 0.1).** YOLO's `val` points to our `val`.

### 3.9 Shared rendering — `wafer/viz.py`
```python
PALETTE = {0: (20, 20, 20), 1: (60, 160, 90), 2: (220, 60, 60)}  # no-die, pass, fail
def render_wafer(map2d: np.ndarray, size: int | None = None) -> np.ndarray: ...  # returns HxWx3 uint8
def render_heatmap(map2d: np.ndarray, heatmap: np.ndarray, bbox=None) -> np.ndarray: ...  # overlay for 8 + dashboard
```
The YOLO PNGs, the dashboard (Phase 6) and pitch figures must all use these functions, so wafers look the same everywhere.

### 3.10 Phase 1 checklist
- [ ] `scripts/download_data.sh` works from a clean machine (with Kaggle env vars); SHA-256 recorded
- [ ] `wafer/data/load.py` loads the pickle; the patch is documented
- [ ] Labelled count = 172,950 (or the real number is logged with an explanation)
- [ ] Both encodings produced; scratch-survival tests (on-grid and offset) pass
- [ ] Duplicates hashed; lots sharing informative maps linked; duplicate, conflict and linking counts logged
- [ ] Grouped split via seed search; all assertions pass; lots per class reported
- [ ] **Thu 10:00:** `wafers_s64.npz`, `metadata.parquet` written → Phase 2 unblocked
- [ ] `yolo_data/`, `wafers_s96.npz`, report figures (non-critical)
- [ ] `docs/results/data_report.md` written
- [ ] `pytest tests/test_data.py` passes
- [ ] Phase 2 and Phase 4 owners told the data is ready (post in team chat + tick in `docs/STATUS.md`)

---

## 4. Phase 2 — Environment on AMD (Anisha & Akein)

### 4.1 PyTorch on ROCm (MI300X on AMD Developer Cloud)
- Use AMD's ROCm PyTorch Docker image (or the ROCm PyTorch wheel). Don't pip-install the default CUDA wheel.
- Sanity check (`scripts/check_gpu.py`):
```python
import torch
print(torch.__version__, "hip:", torch.version.hip)        # hip should be non-None on ROCm
print("gpu available:", torch.cuda.is_available())           # ROCm exposes the GPU through the torch.cuda API
print(torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda"); print((x @ x).sum().item())
```
- In code, always use `device = "cuda" if torch.cuda.is_available() else "cpu"`. On ROCm, `"cuda"` means the AMD GPU. **Never** call CUDA-only libraries (e.g. NVIDIA-specific kernels or `nvidia-smi` in scripts). Use `rocm-smi` to monitor the GPU.
- Environment: `export MIOPEN_FIND_MODE=FAST` (avoids slow first epochs while MIOpen searches for kernels) and `export MIOPEN_USER_DB_PATH=/workspace/.miopen` (a writable cache inside Docker).
- `torch.autocast("cuda", dtype=torch.bfloat16)` for mixed precision; if bf16 causes problems, fall back to fp32 and log it. `torch.compile` is optional. Switch it off at the first error; it isn't worth debugging this week.
- Extra packages: `torch torchvision scikit-learn matplotlib pyyaml pandas tqdm safetensors` (McNemar uses `scipy`, which scikit-learn already installs). One-command setup: `scripts/amd/start_container.sh` then `scripts/amd/bootstrap.sh`.
- Record in the Decisions Log: ROCm version, PyTorch version, GPU name. The pitch says "trained on AMD Instinct MI300X with ROCm", and that must be true.

### 4.2 Data on the GPU
- **No DataLoader.** Load `wafers_s64.npz` once and move `X_area`/`X_nearest`/`y` to the GPU as uint8 tensors (about 2 GB, small next to 192 GB). Each step indexes a batch of rows, converts to float, builds channels and augments, all on the GPU.
- **Parallel runs:** several training processes can share the one MI300X. `scripts/run_sweep.sh` launches N at a time (start with N = 8, and watch `rocm-smi`).

---

## 5. Phase 2 — Model and training

### 5.1 Code layout
```
wafer/
  constants.py        # CLASSES, CONFUSABLE_PAIRS (Section 2.3)
  viz.py              # render_wafer, render_heatmap (Section 3.9)
  data/
    load.py  preprocess.py  split.py
    dataset.py        # GPU-resident tensors; encoding A | B | B+r -> float (B,C,S,S)
    augment.py        # D4, small rotation, die-flip noise (GPU, batched)
    sampler.py        # per-epoch 'none' downsampling + effective-prior bookkeeping
  features.py         # hand-crafted features for E0b
  models/
    small_cnn.py      # team baseline
    resnet.py         # ResNet-18, small-image stem (stride 1 or 2)
    gbdt.py           # E0b classical baseline wrapper
    factory.py        # build_model(cfg)
  train.py            # python -m wafer.train --config configs/e0_baseline.yaml
  evaluate.py         # val by default; --final for test (Section 0.1)
  calibrate.py        # prior correction + temperature + operating point + abstention
  metrics.py          # every metric in Section 7
  explain.py          # Grad-CAM + bbox
  infer.py            # WaferClassifier + module-level classify/explain (Section 9)
configs/              # one YAML per experiment
scripts/              # download_data.sh, prepare_data.py, check_gpu.py, run_sweep.sh,
                      # aggregate.py, bench_latency.py, fetch_model.sh
tests/                # test_data.py, test_model.py, test_infer.py, test_metrics.py
experiments/          # gitignored; one folder per run
docs/results/         # committed: reports, figures, leaderboard.csv (generated), MODEL_CARD.md
```

### 5.2 Input
`WaferDataset(encoding="area"|"nearest"|"area_r", split=..., size=64, device=...)` returns batches `(x, y, uid)`:
- `area` (B, default): `[wafer_frac, pass_frac, fail_frac]` from the stored uint8 /255
- `nearest` (A): one-hot of {0,1,2} → 3 channels {no-die, pass, fail}
- `area_r`: `area` + radius channel → 4 channels (models take `in_chans` from the config)
- Optional auxiliary scalars (E8): `log(dieSize)`, `orig_h/orig_w`, fail fraction. Concatenated after global pooling.

### 5.3 Class imbalance (train only)
- **Per-epoch `none` downsampling** (`sampler.py`): each epoch uses all defect wafers plus a fresh random sample of 15,000 `none` wafers, so the model sees all `none` data over the course of training.
- **Class-weighted loss**: `w_c = (N / n_c) ** 0.5`, normalised to mean 1. The square root stops Near-full from dominating.
- **Effective training prior:** `sampler.py` saves `π_train_eff[c] ∝ n_c_per_epoch · w_c` (sample counts × loss weights) to `experiments/<id>/train_prior.json`. Calibration needs it to undo the shift (7.4).
Never rebalance val or test.

### 5.4 Augmentation (train only, on GPU per batch)
- **D4 symmetries:** random choice of the 8 rotations by 90° × flips. Exact on a square grid.
- **Small rotation:** ±15°, nearest-neighbour, then multiply by the rotated wafer mask so no dies appear outside the wafer. Probability 0.5. Ablate in E3b.
- **Die-flip noise**, defined per encoding: for `nearest`, flip pass↔fail on a fraction `p` of in-wafer pixels; for `area`, at a fraction `p` of in-wafer pixels set `fail_frac ← wafer_frac − fail_frac` (the pass/fail mix at that pixel swaps). `p ∈ [0, 0.005]`. Keep it low so `none` doesn't turn into `Random`. Ablate.
- **Forbidden:** random crops, translations, scaling, colour jitter. Position is part of the label.

### 5.5 Architectures
| ID | Model | Cost at 64×64 (measured, CPU 4 threads, batch 1) | Notes |
|---|---|---|---|
| `small_cnn` | 4 blocks of [Conv3×3–BN–ReLU ×2], max-pool after blocks 1–3 only, channels 32→64→128→256, global average pool, dropout 0.3, FC→9 | 0.21 GMACs · ~3 ms · 1.2M params | Team baseline. Dropping the 4th pool keeps FLOPs the same and gives an 8×8 final map instead of 4×4 |
| `resnet18s` | torchvision `resnet18(weights=None)`, `conv1 = Conv3×3 stride 2`, `maxpool = Identity` | 0.56 GMACs · ~7 ms · 11.2M | Default ResNet |
| `resnet18s1` | Same, `conv1` stride 1 | 2.2 GMACs · ~20 ms (~100 ms with TTA) | **Serve without TTA only.** Include in E3 only if `resnet18s` beats `small_cnn` |
| `gbdt` (E0b) | `HistGradientBoostingClassifier` on `features.py` | CPU, < 1 ms | Classical baseline (Section 6.1) |
| timm backbones | — | — | Only if the GPU is idle and a person is free (E7) |

Every CNN exposes `forward(x) -> logits` and a `cam_layer` attribute: the module whose output is **≥ 16×16** for a 64×64 input (`small_cnn`: block 2 output, before pooling = 32×32, or block 3 output, before pooling = 16×16; `resnet18s`: `layer2` = 16×16; `resnet18s1`: `layer3` = 16×16). The synthetic-insertion benchmark (Section 8) picks the layer.

### 5.6 Training loop (`wafer/train.py`)
- Optimiser AdamW, lr 1e-3, weight decay 1e-4; **cosine schedule over a fixed number of epochs** with 1 epoch linear warm-up.
- Batch size 256. **Epochs: 30 by default.** E0 trains for 60 and logs val curves. If val macro-F1 has plateaued by epoch N, set the default to N for everything else and log it. **No early stopping**: use the final checkpoint. Val metrics are logged every epoch for diagnosis only, never for choosing a checkpoint.
- Loss options: `ce`, `weighted_ce`, `focal` (γ = 2, class weights as α), `ce_ls` (label smoothing 0.05).
- bf16 autocast; gradient clipping at 1.0.
- Seeds: Python, NumPy and Torch seeds set; `torch.backends.cudnn.benchmark = True` is fine (it maps to MIOpen). Log the seed.
- Each run writes `experiments/<run_id>/`: `config.yaml`, `weights.safetensors`, `history.csv` (per epoch: train loss, val loss, val macro-F1, val defect macro-F1, val balanced accuracy, lr), `val_logits.npy` + `val_uids.npy` (for calibration, ensembling and paired tests), `train_prior.json`, `val_metrics.json`, `summary.json` (one leaderboard row) and `env.json` (GPU, ROCm, torch versions, git commit, **dirty flag**).
- `scripts/aggregate.py` rebuilds `docs/results/leaderboard.csv` from every `summary.json`: `run_id, experiment, model, encoding, loss, sampler, aug, seed, val_macro_f1, val_defect_macro_f1, val_bal_acc, val_defect_recall, val_ece_defect, cpu_ms, epochs, train_minutes, git_commit, dirty`. Runs never write to the CSV directly, which avoids merge conflicts.

### 5.7 Config example — `configs/e0_baseline.yaml`
```yaml
experiment: E0
model: small_cnn
encoding: area
size: 64
loss: weighted_ce
sampler: none_downsample   # or: none
none_per_epoch: 15000
aug: {d4: true, small_rot: false, die_flip: 0.0}
optim: {lr: 1.0e-3, wd: 1.0e-4, epochs: 60, batch_size: 256, warmup_epochs: 1}   # E0 only: 60, to read the curve
seeds: [0, 1, 2, 3, 4]
aux_features: false
```
`scripts/run_sweep.sh configs/tier1/*.yaml` runs every config × seed, N processes at a time.

---

## 6. Phase 2 — Experiment plan

Work in tiers. **Don't start a tier until the one before it is done and handed off.**

### 6.1 Tier 0 — must have (Thursday 15 Oct)
| ID | What | Why | Due |
|---|---|---|---|
| E0 | `small_cnn`, encoding B, weighted CE + `none` downsampling, D4 aug, 5 seeds, 60 epochs | Baseline and epoch-count calibration. Calibrate (7.4) and **export as `models/v0`** to Phases 5/6 straight away | Thu 18:00 |
| E0b | Classical baseline: `features.py` (fail fraction; fail density in 5 radial bands and 8 angular sectors; edge-vs-centre ratio; largest fail component area, eccentricity and solidity; Radon-transform row/column statistics) + `HistGradientBoostingClassifier(class_weight="balanced")` | Standard approach in the WM-811K literature, runs in minutes on CPU, and is a pre-registered comparator. Anyone free can own it; it shouldn't block E0 | Thu 22:00 |

### 6.2 Tier 1 — the core comparison (Friday 16 Oct)
**How it runs:** all rows below are **one-factor-at-a-time changes from E0, launched together Friday morning** (`configs/tier1/`), 5 seeds each. Around 15:00, combine the per-row winners into one config (**E-COMB**), run it for 5 seeds, and confirm it beats E0 with a paired lot-bootstrap on val (7.3). If two winners clash (the combination is worse than its parts), drop the factor with the smaller gain.

| ID | Variable | Options |
|---|---|---|
| E1 | Input encoding | B (`area`) vs A (`nearest`) vs B + radius (`area_r`) |
| E2 | Imbalance recipe | weighted CE + downsample (E0) · focal + downsample · CE + label smoothing + downsample · weighted CE, no downsample |
| E3 | Architecture | `small_cnn` vs `resnet18s` (and `resnet18s1` only if `resnet18s` wins) |
| E3b | Augmentation | D4 only vs D4 + small rotation vs D4 + rotation + die-flip |
| E-LEAK | Split protocol | E0 config trained and scored on a **wafer-random** 82/18 split of train+val, vs the same config on the lot-grouped train/val. Report the gap ± seed SD. Never touches test |
| E5 | TTA | Average softmax over the 8 D4 transforms (no retraining). Must fit the latency budget |
| E6 | Ensemble | Average probabilities of the best 3–5 runs. Must fit the latency budget |
| — | Calibration | Section 7.4 on every finalist |

**Selection rule:** the winner of each row has the highest **mean val macro-F1 over 5 seeds**. If two options are within one seed SD, pick the **simpler and faster** one. A config over the latency budget is out (Rule 0.10). Log every choice in the Decisions Log.

**Model freeze: Friday 16 Oct, 21:00.** The best eligible config at that point is calibrated and exported as `models/v1` (9.4). Tier 2 work can replace it only if it beats v1 on val (paired lot-bootstrap, CI excludes 0) **and** passes every test before Saturday 10:00.

### 6.3 Tier 2 — stretch (only after v1 is frozen and handed off). In priority order:
| ID | What | Notes |
|---|---|---|
| E9 | Use the ~638k **unlabelled** maps | **(a) first:** pseudo-labelling. A v1-ensemble teacher labels unlabelled wafers; keep those with calibrated max-prob ≥ 0.98 (cap `none` at 3× the defect count); retrain the student on labelled + pseudo-labelled data. **(b) only if hours remain:** SimCLR-style pretraining with D4 + die-flip views. **Exclude every unlabelled wafer whose `lotName` is in val or test.** This is the most GPU-heavy item and the strongest "why MI300X" story |
| E10 | MixedWM38 stress test (not for training) | Run v1 on MixedWM38 **single-pattern** wafers (control) and **mixed-pattern** wafers. Report the abstention rate on each. Only the **gap** (mixed − single) supports "it knows mixed wafers are out of scope". The control absorbs the dataset shift. Check licence and value encoding first |
| E11 | Unseen wafer geometry | CV on train+val with folds grouped by `(orig_h, orig_w)` instead of lot. Approximates "a new product". Reported as a robustness number, not the headline |
| E8 | Auxiliary scalars | Concatenated after pooling. **Shortcut warning:** these are stand-ins for product. Adopt only if they help on E11 as well as on val |
| E4 | Resolution 96 vs 64 | Uses `wafers_s96.npz`. Check latency |
| E7 | timm backbones | Last |

### 6.4 Optional cross-validation
If GPU time allows, run the final configuration with 5-fold lot-grouped CV over train+val (`cv_fold`) and report mean ± SD macro-F1. Test is never used.

---

## 7. Phase 2 — Evaluation (`wafer/metrics.py`, `wafer/evaluate.py`)

### 7.1 Metrics computed for every run (val) and for the final model (test, once)
**Headline**
- Macro-F1 (9 classes, primary) · defect macro-F1 (8 classes) · balanced accuracy · defect recall and defect precision **at the operating point** (7.4)

**Full picture**
- Per-class precision, recall, F1, support (wafers **and** lots)
- Overall accuracy and weighted F1 (report, never headline)
- MCC, Cohen's kappa, top-2 accuracy
- One-vs-rest ROC-AUC and PR-AUC (macro); PR-AUC is the more informative under imbalance
- Confusion matrix as counts **and** row-normalised (recall view)
- Confusable-pair error rates (`CONFUSABLE_PAIRS`)

**Trustworthiness**
- NLL and multi-class Brier score
- ECE (15 equal-mass bins): overall, **on predicted-defect wafers**, and **classwise**; before and after calibration; reliability diagrams
- Coverage–risk curve and **AURC**, overall and on predicted-defect wafers
- Selective accuracy at 90% / 95% defect coverage
- **Per-class review rate:** fraction of each true class routed to `needs_review`. If 60% of Scratch goes to review, we say so.

### 7.2 Slices (robustness tests)
Report macro-F1 by:
- original wafer size (`orig_h * orig_w` quartiles). Small wafers lose detail when resized
- `dieSize` quartiles
- `waferIndex` (early vs late in lot). A sanity check; should be flat
- D4 consistency: fraction of wafers whose prediction is the same under all 8 symmetries
- Noise robustness: macro-F1 when 0.5% / 1% / 2% of in-wafer dies are flipped
- Sanity inputs: an all-pass wafer → `none`; an all-fail wafer → `Near-full`; an all-zero map → `ValueError` (9.1)

### 7.3 Statistics
- **Bootstrap CIs by lot:** resample `split_group`s (a lot, or lots linked by duplicate maps) with replacement 1,000 times; 95% CI for macro-F1, defect macro-F1, balanced accuracy, defect recall and per-class F1. Resampling lots (not wafers) respects the correlation within a lot. Say plainly that Near-full and Donut CIs are wide.
- **Paired comparisons (primary significance test):** paired lot-bootstrap of Δmacro-F1 and Δdefect-recall between two models on the same wafers (same lot resample for both). Significant ⇔ 95% CI excludes 0. Used for E-COMB vs E0, v2 vs v1, and final vs comparators.
- **McNemar (secondary):** reported with the caveat that it treats wafers as independent (they aren't, within a lot) and tests accuracy, not macro-F1.
- Across seeds: mean ± SD for every Tier 1 row.

### 7.4 Calibration, operating point and abstention — `wafer/calibrate.py`
All fitted on **val**, saved to `models/<version>/calibration.json`.
1. **Ensemble/TTA combine:** average member (and TTA) **probabilities**, then `z = log(p_avg)`. A single model uses `z = logits`.
2. **Prior correction (exact, no fitting):** `z' = z + log π_natural − log π_train_eff`, with `π_natural` = train-split class frequencies and `π_train_eff` from `train_prior.json` (averaged over members). This undoes the rebalancing so probabilities refer to a real fab's class mix. **Temperature scaling alone can't do this**, because it never changes the argmax.
3. **Temperature:** fit a single `T` on `z'` by minimising val NLL (LBFGS). `p = softmax(z' / T)`. These are the probabilities the API returns.
4. **Operating point (defect vs none):** decision: `none` only if `none` is the top class **and** `p(none) ≥ θ_none`; otherwise argmax over the 8 defect classes. `θ_none = 0` is plain argmax, and raising it only turns argmax-`none` wafers into defect calls. Pick the **smallest** `θ_none` whose val defect recall ≥ `R_target` (default 0.95, the hand-off target; a product decision, so raise it to 0.98 if the team prefers fewer misses over fewer false alarms). If argmax already meets the target, `θ_none = 0`: the rule never gives away recall the model already has. Log the precision this gives, and report macro-F1 both at this operating point and at plain argmax.
5. **Abstention:** `τ` = the smallest value such that, on val wafers **predicted as a defect**, those with `confidence ≥ τ` reach selective accuracy ≥ 0.95. Log the coverage and the per-class review rates.
6. **Rule:** `needs_review = True` if any of:
   - `confidence < τ` → reason `"low_confidence"`
   - top-2 classes are a confusable pair and `p1 − p2 < 0.15` (fixed constant, not tuned) → `"confusable_pair"`
   - the decision was a defect but `p(none)` is the single highest probability → `"possible_defect"` (the operating point overrode the argmax)
7. Save `T`, `θ_none`, `τ`, `π_natural`, `π_train_eff`, the margin constant and the val coverage at `τ`.

Calibrating and choosing thresholds on the same val set that's reported makes post-calibration val numbers slightly optimistic. Label them that way; test is the honest number.

### 7.5 Figures and reports (`docs/results/<run_or_final>/`)
- `confusion_counts.png`, `confusion_norm.png`
- `reliability_before.png`, `reliability_after.png` (overall + predicted-defect)
- `coverage_risk.png`, `review_rate_per_class.png`
- `per_class.csv`, `metrics.json`, `slices.csv`
- `error_gallery.png`: the 30 most confident **wrong** predictions, rendered with `render_wafer`, labelled "true / predicted / confidence".
- **Optional blind relabel (val only):** two teammates independently label the 50 most-confident val errors without seeing either label. The share where both disagree with the dataset label is a measured label-noise estimate for the pitch.
- `gradcam_examples.png`: one correct example per class, with heatmap and box

### 7.6 The final test evaluation (once)
```bash
python -m wafer.evaluate --final --model models/v1 \
       --comparators experiments/<e0_best_seed> experiments/<e0b> yolo_preds.csv   # refuses to run without --final
```
- Writes `docs/results/final/` with everything in 7.1–7.5 for v1, plus bootstrap CIs and paired lot-bootstrap comparisons against each comparator. The comparator list is fixed in this command **before** it runs and recorded in the Decisions Log.
- **No evaluation on the dataset's original `trianTestLabel == "Test"` subset.** Our training split contains about 70% of those wafers, so the number would be contaminated. E-LEAK replaces it as the "how do we compare with published protocols" result.
- Generates `docs/results/MODEL_CARD.md` (Section 10).
- **Only numbers in `docs/results/final/` may go in the pitch** (plus E-LEAK and CV numbers, clearly labelled as validation).

---

## 8. Phase 2 — Explainability (`wafer/explain.py`)

YOLO detection can't be trained on WM-811K because it has no bounding boxes. We get localisation through **attribution**. Call it that in the pitch, not "detection".
1. **Grad-CAM** for the predicted class on the model's `cam_layer`, its **last 8×8 stage** (class-specific). ReLU, normalise to [0,1], upsample to the original wafer size.
2. **Bounding box:** keep fail dies whose heat is ≥ 0.25 × peak, group them into clusters (3×3 closing, so a dotted scratch or sparse arc is one cluster), and return the **full** box of the cluster with the most heat, in **original wafer coordinates**. The heatmap *chooses* the cluster; the cluster's own extent sets the box. For `none`, return no box. For `Random`, `Near-full` and `Edge-Ring`, return the wafer's bounding box, since those patterns span the wafer.
3. For an ensemble or TTA, compute Grad-CAM on the single best member (it's for explanation, not scoring).
4. **Synthetic-insertion benchmark** (`scripts/bench_localisation.py`, val wafers only): take 300 val `none` wafers and paste a known synthetic pattern with a known box: a 1-die-wide line (Scratch-like), a filled blob of fail dies (Loc-like), or an arc at the edge (Edge-Loc-like). Report, **on the insertions the model classifies as the intended class**: classification rate, pointing-game hit rate (heatmap max inside the true box), and median box IoU. Use it to choose `cam_layer`. Synthetic patterns aren't the real distribution, so these are sanity numbers, not headline numbers.

---

## 9. Interfaces for other phases (contract — don't change without Section 0.8)

### 9.1 `wafer/infer.py`
```python
# Object API (preferred)
from wafer.infer import load_model
clf = load_model("models/v1", device="cpu")   # -> WaferClassifier; loads weights + calibration.json
clf.classify(wafer_map) -> dict
clf.classify_batch(maps) -> list[dict]
clf.explain(wafer_map) -> dict

# Module-level convenience (unchanged call sites for Phases 5/6):
from wafer.infer import classify, classify_batch, explain
# These use a lazily loaded default model: $WAFER_MODEL_DIR, else "models/v1".
```
`classify` returns **only plain Python types** (str, float, bool, list, dict), so the agent can JSON-serialise it directly:
```python
{
  "pattern": "Edge-Ring",                # decision per 7.4 step 4 (always present; Phase 5/6 rely on this key)
  "confidence": 0.94,                    # calibrated probability of "pattern"
  "probs": {"Center": 0.01, ...},        # all 9 calibrated probabilities, sum to 1
  "top2": [["Edge-Ring", 0.94], ["Donut", 0.03]],
  "defect_prob": 0.97,                   # 1 - probs["none"]
  "needs_review": False,                 # 7.4 step 6; the agent must say it's unsure when True
  "review_reasons": [],                  # subset of ["low_confidence", "confusable_pair", "possible_defect"]
  "model_version": "v1"
}
```
```python
explain(wafer_map) -> {
  "heatmap": np.ndarray,                 # float32, same HxW as input, values in [0,1]
  "bbox": [x0, y0, x1, y1] | None,       # original coordinates; None for "none"
  "pattern": "Edge-Ring"
}
```
- Preprocessing inside `classify` calls **the same functions** as Phase 1 (`wafer/data/preprocess.py`). It must not be re-implemented.
- **Latency budget:** < 50 ms per wafer for the served config, measured by `scripts/bench_latency.py` with `torch.set_num_threads(4)`, batch 1, after 10 warm-up calls, mean over 200 calls. Record the CPU model in `models/<version>/README.md`.
- Never raises on unusual input sizes. On invalid input (not 2D, values outside {0,1,2}, no dies at all), raise `ValueError` with a clear message.
- Phase 5's `classify_wafer` tool wraps `classify()` directly. Phase 6 uses `classify()`, `explain()`, `render_wafer()` and `render_heatmap()`.

### 9.2 Mock for early integration
By **Thu 15 Oct 12:00**, provide `wafer/infer_mock.py` with the same signatures and output keys (including `defect_prob` and `review_reasons`), returning plausible fixed outputs. Phases 5 and 6 build against it and swap the import later.

### 9.3 Data contract for Phase 4
`data/processed/metadata.parquet` (Section 3.8). Phase 4 builds `stream.csv` from **`split == "val"` wafers until the final test evaluation has run (Sat 10:00), then rebuilds it from `split == "test"`**, so the demo shows unseen wafers without anyone looking at test predictions beforehand. Phase 4 may group wafers into fictional lots; keep `uid` so each streamed wafer can be traced back.

### 9.4 Model release
`models/<version>/` contains `weights/*.safetensors` (one per member), `config.yaml`, `calibration.json`, `classes.json`, `metrics_val.json`, `README.md` (served config, CPU latency and the CPU it was measured on).
- If the folder is < 50 MB, commit it. Otherwise upload it as a GitHub Release asset and add `scripts/fetch_model.sh`.
- `v0` = E0 (Thursday). `v1` = freeze (Friday 21:00). `v2` only through the freeze rule in 6.2.

### 9.5 Benchmarking other models fairly (Phase 3, E0b)
`wafer/evaluate.py --predictions path/to/preds.csv` accepts a CSV of `uid, prob_<class>...` covering **val and test** uids, and scores it with **exactly the same metrics, splits and statistics** as the CNN. Raw (uncalibrated) probabilities are fine. `evaluate.py` applies the same 7.4 procedure, fitted on that model's own **val** rows, so every model gets the same calibration and operating point. If the model's training prior isn't known, step 2 (prior correction) is skipped and logged.

---

## 10. Model card — `docs/results/MODEL_CARD.md` (generated, then edited by hand)
Sections: intended use (assist yield engineers; not an autonomous decision-maker) · training data, split method and the E-LEAK result · architecture and training details · hardware (AMD Instinct MI300X, ROCm version) · final test metrics with 95% lot-bootstrap CIs · operating point, calibration and abstention behaviour, with per-class review rates · localisation sanity numbers (Section 8.4) · known limitations (label noise with the measured floor; single-label only; one fab's data; product shortcut risk with the E11 result if run; small classes have wide CIs; mixed-pattern wafers are out of scope and should be flagged) · how to reproduce.

---

## 11. Tests (`tests/`, run with `pytest -q`; CI optional)
**`test_data.py`**
- loader returns a dataframe with the expected columns; labelled count matches the data report
- `_flat` handles `[['x']]`, `['x']`, `[]`
- resize: shapes, value sets, `pass_frac ≥ 0`, scratch survival on-grid **and** one-die-offset (3.5)
- split: no lot overlap; `MIN_COUNT` met; relative deviation within tolerance; `cv_fold ∈ {-1,0..4}` and `-1` exactly on test
- `informative`, `link_lots` (chains merge; uniform maps never link) and the cross-split duplicate check; on the synthetic set, planted duplicates share the original's split and nothing is dropped
- npz rows align with metadata `uid`s

**`test_model.py`**
- every CNN in the factory maps `(B,C,64,64)` → `(B,9)` and its `cam_layer` output is 8×8
- one training step on a tiny batch reduces loss (smoke test)
- augmentation keeps the wafer mask and the label unchanged; D4 transforms are exact (applying all 8 and inverting returns the original); die-flip keeps `pass_frac ≥ 0`
- `train_prior.json` sums to 1 and matches a hand computation on a toy sampler

**`test_metrics.py`**
- metrics on a hand-made example match hand-computed values
- ECE is 0 for a perfectly calibrated toy case
- bootstrap-by-lot returns intervals containing the point estimate; paired bootstrap of a model against itself gives a Δ CI containing 0
- prior correction: with `π_train_eff = π_natural` it is the identity
- operating point: on a toy case, the chosen `θ_none` gives the stated recall

**`test_infer.py`**
- `classify` returns all keys with correct **plain Python** types; `probs` sums to 1 ± 1e-5; `json.dumps` succeeds
- works on maps of size 10×10, 64×64, 300×200
- invalid input raises `ValueError`
- **train/serve parity:** for 200 val uids, `classify(raw_map)["probs"]` equals the `evaluate.py` pipeline's probabilities to 1e-5
- CPU latency smoke test < 150 ms (the real budget is enforced by `scripts/bench_latency.py`; this looser bound keeps pytest from failing on slow laptops)
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
| 14 Oct | review | Test scored once, on one model; comparators pre-registered, never selected on test | "Once per candidate" amounts to model selection on test |
| 14 Oct | review | Drop evaluation on original `trianTestLabel == "Test"`; add E-LEAK | ~70% of those wafers are in our train split; E-LEAK measures the leakage gap honestly |
| 14 Oct | review | Latency is a hard eligibility gate; `resnet18s` uses a stride-2 stem; new Rule 0.10 | Measured: stride-1 ResNet-18 = 20 ms/wafer, ~670 ms with 5 members × TTA (budget 50 ms) |
| 14 Oct | review | Grad-CAM on a ≥ 16×16 layer; `small_cnn` drops its 4th pool; synthetic-insertion benchmark | v1's last-layer CAM was 4×4 (`small_cnn`) / 8×8 (`resnet18s`), too coarse for scratches |
| 14 Oct | review | Exact prior correction before temperature scaling | Rebalanced training shifts priors; temperature can't change argmax, so it can't fix that |
| 14 Oct | review | Explicit operating point `θ_none` for defect recall; two-stage decision | Ties the defect-recall target to an actual mechanism and to the fab's costs |
| 14 Oct | review | Targets and τ defined on predicted-defect wafers; classwise ECE; per-class review rate | Overall ECE/selective accuracy pass trivially with 85% easy `none` |
| 14 Oct | review | Fixed-length cosine, no early stopping; 5 seeds | Early stopping cuts cosine at high LR and tunes on noisy val macro-F1; runs are cheap |
| 14 Oct | review | Paired lot-bootstrap replaces McNemar as primary test | McNemar ignores lot correlation and tests accuracy, not macro-F1 |
| 14 Oct | review | Encoding B is the default from E0 | Measured: a one-die-offset scratch disappears completely under nearest (0/140 pixels) |
| 14 Oct | review | ~~Drop cross-split dupes only if `n_fail ≥ 1`~~ (superseded below); report conflicting-label dupes as noise floor | Identical all-pass maps are different wafers; dropping them biases eval |
| 14 Oct | review | Automatic split-seed search with minimum per-class counts; report lots per class | Rare classes in few lots can vanish from val (simulated worst case: Near-full 0 in val) |
| 14 Oct | review | Fix `cv_fold` to 0–4 over train+val | v1 said 0–13 and contradicted its own CV description |
| 14 Oct | review | GPU-resident data, no DataLoader; parallel runs; per-run `summary.json` + aggregator | Data is ~2 GB; avoids worker issues and leaderboard merge conflicts |
| 14 Oct | review | Parallel one-factor sweep + combination run instead of sequential Tier 1 | Removes ~6 serial human round-trips on the day before freeze |
| 14 Oct | review | Drop `resnet18s_pt`; add classical baseline E0b; promote E9 (pseudo-labels first); add E11; MixedWM38 needs a single-pattern control | Better use of time; more honest comparisons |
| 14 Oct | review | `load_model` returns `WaferClassifier`; output adds `defect_prob`, `review_reasons`; plain types; train/serve parity test | v1 API was ambiguous about which model `classify` used; additive change, call sites unchanged |
| 14 Oct | review | Demo stream from val until final eval, then test | Dashboard development would otherwise show test predictions for days |
| 14 Oct | review | Phase 1 critical path = npz + parquet by Thu 10:00; YOLO PNGs off critical path | Phase 2 doesn't need PNGs |
| 14 Oct | Phase 1 impl | **Link lots that share an informative duplicate map into one `split_group` before splitting, instead of dropping duplicates after.** Informative = `min(n_fail, n_pass) ≥ 1` | Real-scale synthetic run: dropping after the split pushed Near-full test below its minimum (12 < 15) after the seed search had already passed it. All-fail maps also repeat between different wafers. Linking loses no wafers and keeps the search honest |
| 14 Oct | Phase 1 impl | Bootstrap and CV by `split_group` | Linked lots are not independent |
| 14 Oct | Phase 1 impl | `uid` = positional row number in the raw dataframe | Stable regardless of the pickle's index |
| 14 Oct | Phase 1 impl | Labelled rows that aren't 2D, have values outside {0,1,2}, or have no dies are dropped and counted in the report | `infer` rejects such maps anyway (9.1); keeping them would poison training |
| 14 Oct | Phase 1 impl | `requirements.txt` uses bounded versions; exact versions recorded in the data report | Exact pins can't install on both Python 3.10 (ROCm image) and 3.13 |
| 14 Oct | Phase 1 impl | Split search runs in a `spawn` process pool | Forking a process that already runs BLAS threads can deadlock |
| 14 Oct | Phase 2 impl | AMD workflow = `scripts/amd/{start_container,bootstrap,run_e0}.sh`; results come back to the repo as committed `docs/results/` + `models/vN/` | All training/eval must run on the AMD cloud; the code is written and tested on CPU elsewhere |
| 14 Oct | Phase 2 impl | MIOpen env defaults (`MIOPEN_FIND_MODE=FAST`, repo-local `MIOPEN_USER_DB_PATH`) are set in code (`wafer/env.py`) | Nobody has to remember to export them |
| 14 Oct | Phase 2 impl | E8 aux scalars use `log(n_die)`, log aspect ratio and fail fraction, all computed from the map, instead of `log(dieSize)` | `classify()` only receives the map; train/serve parity |
| 14 Oct | Phase 2 impl | `cam_layer` is a property, not a registered submodule | Registering it duplicated weights in the state dict (safetensors refuses to save) |
| 14 Oct | Phase 2 impl | `models/v0` = the E0 seed with the best val macro-F1; export recomputes val outputs exactly as served before fitting calibration | v0 is an integration build; recomputing guarantees calibration matches serving (incl. TTA) |
| 14 Oct | Phase 2 impl | Module-level `classify()` default: `$WAFER_MODEL_DIR`, else `models/v1`, else `models/v0` | Phases 5/6 work as soon as v0 exists |
| 14 Oct | Phase 2 impl | `infer_mock` returns heuristic (geometry-rule) outputs, not fixed ones | Lets the dashboard/agent exercise varied patterns and `needs_review` paths |
| 14 Oct | Phase 2 impl | Operating point can only *add* defect calls to argmax (`none` needs top class AND `p(none) ≥ θ`); θ = 0 when argmax already meets the recall target | Synthetic run: "smallest θ meeting 95%" lowered recall from 100% to 95% and macro-F1 from 1.000 to 0.977 |
| 14 Oct | Phase 2 impl | GPU may be AMD Radeon (Token Factory = Radeon Cloud), not MI300X. Code is GPU-agnostic on ROCm; every "MI300X" in this spec means "the AMD GPU in `env_amd.json`" | Pitch claims must match the recorded hardware |
| 14 Oct | Phase 1 real run | `LSWMD.pkl` is a **Python 2** pickle: after aliasing `pandas.indexes`, load with pandas' compat `Unpickler(encoding="latin1")` (pandas 3.0.6 / Python 3.13; also tested on pandas 2.1) | pandas 3 raises `UnicodeDecodeError` instead of retrying latin1 |
| 14 Oct | Phase 1 real run | Linked split groups holding > 1% of wafers are pinned to **train** (`--pin-train-frac`) | Real data: 3,234 informative maps appear identically in two lots and chain 598 lots (12,596 wafers, all one 26×26 product) into one group, which otherwise landed whole in test (half of it). Leakage stays blocked; val/test keep a natural product mix. Split is now 72/14/14 |
| 14 Oct | Phase 1 real run | Dataset downloaded without credentials via Kaggle's public download endpoint (redirects to Kaggle's own GCS bucket); SHA-256 `1d04fccb…5c65` recorded in the data report | No Kaggle token in Claude's environment; dataset is public |
| 14 Oct | v0 | `models/v0` trained on **CPU** (`configs/e0_cpu.yaml`: 1 seed, 30 epochs, bf16 + channels_last) | Token Factory has no GPU instances. v0 must not be described as trained on AMD; retrain with `e0_baseline.yaml` on an AMD GPU when one is available |
| 14 Oct | E-LEAK result | **No measurable leakage inflation**: wafer-random split val macro-F1 0.9018 vs lot-grouped 0.9016 (raw argmax 0.859 vs 0.855; 1 seed each, CPU, CI half-width ~0.015). The pitch must not claim "lot grouping is worth L points"; the honest line is "we tested for lot leakage and found none: the score holds under both protocols". Lot grouping stays (it costs nothing and is the right protocol) | Real-data result overturns a spec assumption |
| 14 Oct | E1 result | Nearest (A) vs area-fraction (B): val macro-F1 0.906 vs 0.902, paired Δ +0.005 (CI −0.005 to +0.015, not significant); Scratch F1 identical (0.801). Keep B as default (no evidence against it; it protects the >64-die wafers in theory) | Median wafer is ~33×33 dies, so most maps are *upsampled* to 64 and nearest loses nothing. The scratch-erasure risk only applies to the large-wafer minority |
| 14 Oct | Screening results | E3b rotation +0.007 (n.s.), E2 focal +0.001 (n.s.), **E5 TTA +0.009 (paired CI +0.005 to +0.014, significant)** at 27 ms/wafer CPU. v1 candidate: E0 (+ rotation if 5 seeds confirm) served with TTA | Single-seed CPU screening; confirm with 5 seeds on an AMD GPU |
| 14 Oct | Seeds | E0 vs E3b with 3 seeds each (CPU): 0.9055 ± 0.0040 vs 0.9020 ± 0.0053, so rotation is dropped; seed-0's +0.007 was noise. Seed SD ≈ 0.004–0.005 macro-F1 | Multi-seed check before adopting a change (spec 6.2 selection rule) |
| 14 Oct | E6 result | 3-seed E0 ensemble (`models/v0-ens3`, no TTA): val 0.912, +0.010 vs v0 (significant), ties v0-tta at 9.5 vs 27 ms/wafer. Leading v1 candidate | Ensembles beat TTA on latency for the same gain |
| 15 Oct | Explainability | Box = full fail-die cluster chosen by Grad-CAM heat (≥ 0.25 × peak), CAM on the last **8×8** stage (was: hot region ∩ fails on a ≥16×16 layer); Edge-Ring gets the wafer box; overlay alpha scales with heat | Synthetic insertions (300 each): box IoU ≥ 0.3 went from 98/73/37% to **97/99/100%** (Scratch/Loc/Edge-Loc). On real wafers, the 16×16 layer's heat often sat on the wrong cluster; the 32×32 layer was worse (IoU ≥ 0.3: 46/8/95%) |
| 15 Oct | E3 result | ResNet-18 (stride-2 stem, CPU, 1 seed): val 0.905 vs small CNN 3-seed mean 0.906 (paired Δ +0.003, CI −0.012 to +0.016, n.s.); 4× slower to train, ~6× slower per wafer. Adding it to the 3-CNN ensemble *lowers* val macro-F1 (0.910 vs 0.912). Dropped | Higher raw score (0.869 vs 0.855) disappears after calibration |
| 14 Oct | Phase 2 impl | Noise-robustness slice flips pixels of the 64×64 encoding, not raw dies | Evaluation works from the npz; documented as pixel-level |
| | | | |

---

## 13. Status board — `docs/STATUS.md`
Keep a short checklist in `docs/STATUS.md` (one line per item from 3.10, 6.1–6.3, 7.6, 9.1–9.4) with owner and ✅/🟡/❌. Update it in every PR. It's how teammates and their agents see what's done without asking.

## 14. Timeline (Sydney time)
| When | Milestone |
|---|---|
| Wed 14 Oct night | Repo scaffold, `constants.py`, `viz.py`, GPU check passes, data downloaded + SHA-256 recorded |
| Thu 15 Oct 10:00 | **Phase 1 critical path done:** `wafers_s64.npz` + `metadata.parquet` |
| Thu 15 Oct 12:00 | `infer_mock.py` available to Phases 5/6 |
| Thu 15 Oct 16:00 | Phase 1 non-critical items (PNGs, s96, report figures) |
| Thu 15 Oct 18:00 | E0 trained (5 seeds), calibrated, exported as `v0`; epoch count fixed |
| Thu 15 Oct 22:00 | E0b classical baseline on val; Tier 1 configs written and dry-run |
| Fri 16 Oct 09:00 | Tier 1 sweep launched (all rows in parallel) |
| Fri 16 Oct 15:00 | Row winners picked; E-COMB launched |
| Fri 16 Oct 21:00 | **Freeze:** best eligible config calibrated and exported as `v1` |
| Sat 17 Oct 10:00 | Final test evaluation run once; `docs/results/final/` + model card committed; Phase 4 swaps stream to test |
| Sat 17 Oct 19:00 | Team submission deadline (team plan). Official deadline: Mon 19 Oct 02:00 Sydney |

## 15. Pitch-ready claims checklist
Before any number goes on a slide, confirm:
- [ ] it comes from `docs/results/final/` (test, run once), or is explicitly labelled as validation/CV/E-LEAK
- [ ] it has a 95% CI (bootstrap by lot) where relevant
- [ ] the slide says the split is lot-grouped, and quotes the E-LEAK gap
- [ ] any comparison to published numbers says they used different splits and are not directly comparable
- [ ] defect-class results are shown, not just overall ones (no "99% accurate" driven by `none`)
- [ ] localisation is called attribution, with the synthetic-benchmark numbers if quoted
- [ ] the AMD hardware/software claim matches `env.json`
