# Status board

One line per item from spec Sections 3.10, 6.1–6.3, 7.6 and 9.1–9.4. ✅ done · 🟡 in progress / partly done · ❌ not started. Update it in every PR.

## Phase 1 — data (Nasser & Jina)
| Item | Owner | Status | Notes |
|---|---|---|---|
| `scripts/download_data.sh` | Nasser / Jina | ✅ | Works with `KAGGLE_API_TOKEN`, legacy key, or no credentials (public archive) |
| `wafer/data/load.py` + pickle compat patch | Nasser / Jina | ✅ | Real file needs the `pandas.indexes` alias **and** latin1 (Python 2 pickle); logged |
| Clean (3.4) | Nasser / Jina | ✅ | 811,457 rows → 172,950 labelled (0 invalid, 0 empty) |
| Encodings A + B, scratch-survival tests (3.5) | Nasser / Jina | ✅ | |
| Duplicates → lot linking (3.6) | Nasser / Jina | ✅ | 3,250 duplicate groups, 14 with conflicting labels; giant linked group pinned to train |
| Grouped split with seed search (3.7) | Nasser / Jina | ✅ | Seed 98, 72/14/14, worst class-share deviation 7.5%; Near-full in 137 lots |
| **`wafers_s64.npz` + `metadata.parquet`** | Nasser / Jina | ✅ | Built in Claude's environment (7 min). Not in git: rebuild anywhere with `scripts/prepare_data.py` (same SHA-256 → same split) |
| `docs/results/data_report.md` | Nasser / Jina | ✅ | Committed |
| `wafers_s96.npz`, `yolo_data/` (non-critical) | Nasser / Jina | ❌ | `--sizes 64 96 --yolo` |
| `wafer/viz.py` (`render_wafer`, `render_fractions`, `render_heatmap`) | Nasser / Jina | ✅ | |
| `pytest tests/test_data.py` | Nasser / Jina | ✅ | All pass, including the real-data tests |

## Phase 2 — classifier (Anisha & Akein)
| Item | Owner | Status | Notes |
|---|---|---|---|
| AMD container + bootstrap + `scripts/check_gpu.py` | Anisha / Akein | 🟡 | Scripts written; not yet run on the MI300X. Commit `docs/results/env_amd.json` when it passes |
| `wafer/infer_mock.py` | Anisha / Akein | ✅ | Same signatures and keys as `wafer.infer`; heuristic outputs |
| Training core (GPU-resident data, augmentation, sampler, models, losses, `train.py`) | Anisha / Akein | 🟡 | Code + tests ✅ on CPU/synthetic · first MI300X run ❌ |
| Calibration, operating point, abstention (`calibrate.py`, `decision.py`) | Anisha / Akein | ✅ | Tested on toy data |
| `metrics.py`, `evaluate.py` (val; `--final` once) | Anisha / Akein | 🟡 | Done except `--predictions` CSV for YOLO/E0b comparators |
| `wafer/infer.py` + `wafer/explain.py` + `wafer/export.py` | Anisha / Akein | 🟡 | Code + tests ✅ (train/serve parity) · no real model yet |
| E0 baseline → `models/v0` | Anisha / Akein | 🟡 | Training on **CPU** in Claude's environment (no AMD GPU available yet); AMD retrain pending a GPU |
| E0b classical baseline | anyone free | ❌ | |
| Tier 1 sweep + E-LEAK + E-COMB | Anisha / Akein | ❌ | Configs not written yet |
| Freeze `models/v1` | Anisha / Akein | ❌ | |
| Final test evaluation, once | Anisha / Akein | ❌ | |
| Tier 2 (E9, E10, E11, E8, E4, E7) | Anisha / Akein | ❌ | |
