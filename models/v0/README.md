# Model v0

- Members: 1 × small_cnn · encoding `area` · 64px · TTA: off
- Source runs: e0_cpu_s0_20261014-132410
- Calibration (fitted on val): T = 1.188 · θ_none = 0.5200 (defect recall target 0.95) · τ = 0.5847 (selective accuracy target 0.95, defect coverage 0.940)
- **CPU latency (served config, `classify`, batch 1, 4 threads): 4.0 ms** on `x86_64` — budget 50 ms (spec 9.1)

## Validation (optimistic: calibrated on the same set)
| metric | value |
|---|---|
| macro-F1 (9) | 0.9016 (95% CI 0.8847–0.9133) |
| defect macro-F1 (8) | 0.8903 |
| balanced accuracy | 0.8979 |
| defect recall / precision | 0.9501 / 0.9609 |
| ECE on predicted defects | 0.0102 |
| review rate | 0.0109 |

Never quote these as test numbers. Test is scored once, by `python -m wafer.evaluate --final`.
