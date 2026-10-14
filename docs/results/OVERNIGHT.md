# Overnight report — Wed 14 Oct 2026

**TL;DR:** Phase 1 ran on the real WM-811K. A first CNN, `models/v0`, is trained, calibrated, evaluated on validation and committed: **val macro-F1 0.902 (95% CI 0.885–0.913), defect recall 0.95 at 0.96 precision, 4 ms/wafer on CPU.** Serving it with 8-way rotation/flip averaging (`models/v0-tta`) gives **0.911** at 27 ms/wafer. The CNN beats a strong classical baseline by a significant margin. Two spec assumptions turned out wrong on real data, and both are logged.

> **Everything here was trained on CPU** (Claude's cloud environment, 4-core Xeon, bf16), because Token Factory has no GPU instances. **Do not say "trained on AMD" about these models.** All numbers are **validation**: calibration and thresholds were fitted on the same set, so they're slightly optimistic. The test split is still untouched.

---

## 1. Phase 1 on the real data (`docs/results/data_report.md`)
- 811,457 wafers, **172,950 labelled** (exactly as expected), 0 invalid or empty maps. File SHA-256 `1d04fccb…5c65` recorded.
- **Two fixes the real file forced:**
  1. `LSWMD.pkl` is a **Python 2 pickle**. It needs pandas' compat unpickler with `encoding="latin1"` as well as the old `pandas.indexes` alias.
  2. **3,234 wafer maps appear identically in two different lots.** These are the same wafers recorded twice, not coincidences: they have 50+ fail dies each. Linking them chained **598 lots (12,596 wafers, all one 26×26 product)** into one group, which landed entirely in test. Groups that big are now **pinned to train**, so test isn't half one product.
- **Split:** lot-grouped, 72/14/14, every class within 7.5% of its overall share in val and test. **Rare classes are well spread:** Near-full comes from 137 lots and Donut from 233, so their confidence intervals are usable.
- **Label noise floor:** 14 duplicate maps carry conflicting labels.

## 2. Results (validation, single seed, CPU)

| Model | Macro-F1 (9) | Defect macro-F1 | Defect recall / precision | CPU ms/wafer | vs v0 (paired lot bootstrap) |
|---|---|---|---|---|---|
| **v0**: small CNN, area encoding, weighted CE + `none` downsampling, D4 aug (`models/v0`) | **0.902** (0.885–0.913) | 0.890 | 0.950 / 0.961 | 4.0 | — |
| **v0-tta**: same weights, D4 ×8 test-time averaging (`models/v0-tta`) | **0.911** (0.894–0.923) | 0.901 | 0.950 / 0.963 | 27.3 | **+0.009 (+0.005 to +0.014), significant** |
| E0b: classical features + gradient boosting | 0.865 (0.847–0.878) | 0.850 | 0.950 / 0.881 | <1 | **−0.037 (−0.052 to −0.023), significant** |
| E1: nearest-neighbour encoding (team plan) | 0.906 | 0.896 | 0.950 / 0.953 | 2.9* | +0.005 (−0.005 to +0.015), n.s. |
| E3b: + small rotation augmentation | 0.908 | 0.898 | 0.950 / 0.959 | 2.6* | +0.007 (−0.001 to +0.016), n.s. |
| E2: focal loss | 0.902 | 0.891 | 0.950 / 0.959 | 2.5* | +0.001 (−0.010 to +0.009), n.s. |
| E-LEAK: same as v0, but **wafer-random** split | 0.902 | 0.891 | 0.950 / 0.953 | — | different val set: compare 0.9018 vs 0.9016 |

\* model forward pass only; v0's 4.0 ms is the full `classify()` call.

**v0 in detail** (`docs/results/v0_val/`, with confusion matrix, reliability, coverage-risk and error-gallery PNGs):
- Per-class F1: Edge-Ring 0.98 · Near-full 0.97 · Center 0.95 · Donut 0.88 · Random 0.88 · Edge-Loc 0.87 · Scratch 0.80 · Loc 0.79 · none 0.99.
- **Main error: missed defects, not mixed-up patterns.** Scratch is called `none` 14% of the time, Loc 10%, Edge-Loc 9%. Confusable pairs are rare (Edge-Loc↔Loc 3%).
- **Trust:** calibration error on predicted defects is 0.010. 1.1% of wafers are routed to review (Random 12%, Loc 9%, `none` 0.4%). Selective accuracy on the 90% most-confident defect calls is 0.964.
- **Robustness:** 98% of predictions are identical under all 8 rotations/flips; with 2% of dies flipped, macro-F1 only drops 0.902 → 0.873.
- **Weak spot:** small wafers. The smallest quartile (≤754 dies) scores 0.79 versus 0.90 for the largest.
- **Sanity inputs:** all-pass → `none` ✓, all-fail → `Near-full` ✓, empty map → `ValueError` ✓.
- **Calibration matters:** the model's raw top prediction scores 0.855. Undoing the class rebalancing (prior correction) plus temperature brings it to 0.902.

## 3. What the experiments say
1. **The CNN is worth it:** +0.037 macro-F1 over the classical baseline, significant. The biggest gains are on Loc, Scratch and Edge-Loc, the shape-based patterns.
2. **E-LEAK: no measurable leakage inflation** (0.9018 random vs 0.9016 lot-grouped). The spec expected random splits to inflate scores; on this data they don't. **Pitch consequence:** drop "lot grouping is worth L points". Say instead: "we tested for lot leakage: the score holds under both protocols." Lot grouping stays, since it's the right protocol and costs nothing.
3. **E1: the encoding doesn't matter on real data.** Nearest and area-fraction tie, and Scratch F1 is identical. The median wafer is only ~33×33 dies, so most maps are *upsampled* to 64×64, and nearest loses nothing when upsampling. Area-fraction stays the default because it protects the minority of large wafers.
4. **E3b (rotation) does not help.** It looked like the best change on seed 0 (+0.007), so I ran 2 more seeds of both configs: **E0 0.9055 ± 0.0040 vs E3b 0.9020 ± 0.0053** (3 seeds each); on seeds 1 and 2 rotation was slightly worse. Single-seed differences under ~0.01 are noise here, which is exactly why the spec asks for 5 seeds. **E2 (focal loss) does nothing.**
5. **TTA gives a significant +0.009** (paired CI +0.005 to +0.014; McNemar p = 0.0004) for 7× the latency (27 ms), still within the 50 ms budget. It's the one change so far that's clearly worth shipping.

All of this is in the spec's Decisions Log.

## 4. What I'd do next
- **Get an AMD GPU** (AMD Developer Cloud credits, or a teammate's Radeon). Then run `notebooks/amd_e0.ipynb` or `scripts/amd/run_e0.sh`: 5 seeds of E0, plus E3b with 5 seeds, in parallel, in minutes. That gives the honest "trained on AMD" claim and turns the single-seed hints into real comparisons.
- **v1 candidate:** plain E0 served with TTA (27 ms, inside budget), or an ensemble (see §2b). Rotation is out. Freeze it, then run the **one** final test evaluation: `python -m wafer.evaluate --final --model models/v1 --comparators models/v0 experiments/e0b/predictions.csv`. (Regenerate the E0b CSV first, since `experiments/` isn't in git.)
- **Small-wafer weakness:** the smallest quartile is the biggest gap. Worth one experiment (e.g. aux die-count feature E8, or 96 px).

## 5. For the agent and dashboard teammates
`models/v0` is in the repo (9 MB) and works today:
```python
from wafer.infer import classify, explain        # picks models/v1 if present, else models/v0
classify(wafer_map)   # {"pattern", "confidence", "probs", "top2", "defect_prob", "needs_review", "review_reasons", "model_version"}
```
Or `WAFER_MODEL_DIR=models/v0-tta` for the TTA version. Same API as `wafer/infer_mock.py`.

## 6. How to reproduce
```bash
pip install -r requirements.txt torch torchvision
scripts/download_data.sh                       # no Kaggle credentials needed (public archive)
python scripts/prepare_data.py --jobs 4        # ~7 min; same SHA-256 -> same split (seed 98)
python -m wafer.train --config configs/e0_cpu.yaml        # ~55 min on 4 cores
python -m wafer.export --runs experiments/e0_cpu_s0_* --version v0 && python -m wafer.evaluate --model models/v0
```
Screening configs: `configs/cpu/`. Comparison table: `scripts/compare_runs.py` → `docs/results/screening_vs_e0.md`. All runs: `docs/results/leaderboard.csv`.
