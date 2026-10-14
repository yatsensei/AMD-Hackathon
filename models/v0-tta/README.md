# Model v0-tta

- Members: 1 × small_cnn · encoding `area` · 64px · TTA: D4 ×8
- Source runs: e0_cpu_s0_20261014-132410
- Calibration (fitted on val): T = 1.158 · θ_none = 0.5599 (defect recall target 0.95) · τ = 0.5349 (selective accuracy target 0.95, defect coverage 0.952)
- **CPU latency (served config, `classify`, batch 1, 4 threads): 27.3 ms** on `x86_64` — budget 50 ms (spec 9.1)

## Validation (optimistic: calibrated on the same set)
| metric | value |
|---|---|
| macro-F1 (9) | 0.9108 (95% CI 0.8940–0.9226) |
| defect macro-F1 (8) | 0.9006 |
| balanced accuracy | 0.9065 |
| defect recall / precision | 0.9501 / 0.9627 |
| ECE on predicted defects | 0.0091 |
| review rate | 0.0074 |

Never quote these as test numbers. Test is scored once, by `python -m wafer.evaluate --final`.
