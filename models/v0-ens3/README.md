# Model v0-ens3

- Members: 3 × small_cnn · encoding `area` · 64px · TTA: off
- Source runs: e0_cpu_s0_20261014-132410, e0_cpu_s1_20261014-183954, e0_cpu_s2_20261014-201408
- Calibration (fitted on val): T = 1.122 · θ_none = 0.5176 (defect recall target 0.95) · τ = 0.5722 (selective accuracy target 0.95, defect coverage 0.948)
- **CPU latency (served config, `classify`, batch 1, 4 threads): 9.5 ms** on `x86_64` — budget 50 ms (spec 9.1)

## Validation (optimistic: calibrated on the same set)
| metric | value |
|---|---|
| macro-F1 (9) | 0.9117 (95% CI 0.8959–0.9224) |
| defect macro-F1 (8) | 0.9017 |
| balanced accuracy | 0.9102 |
| defect recall / precision | 0.9501 / 0.9617 |
| ECE on predicted defects | 0.0047 |
| review rate | 0.0091 |

Never quote these as test numbers. Test is scored once, by `python -m wafer.evaluate --final`.
