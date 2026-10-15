# Model card — wafer defect-pattern classifier

**Status: DRAFT.** The v1 model is not frozen yet, and the **test numbers are pending** the single final evaluation (`python -m wafer.evaluate --final`). Everything below marked *validation* is slightly optimistic, because calibration was fitted on the same set.

## Intended use
- **Users:** yield and process engineers in a semiconductor fab.
- **Task:** given one wafer bin map (0 = no die, 1 = pass, 2 = fail), name its defect pattern (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none). It also returns a calibrated confidence, a `needs_review` flag with reasons, and a heatmap plus box showing which fail dies drove the decision.
- **It assists, it doesn't decide.** The agent recommends actions (hold lots, open a ticket) and an engineer approves them. Low-confidence or ambiguous wafers are routed to a person.
- **Out of scope:** wafers with more than one pattern at once (single-label model), fabs and products very different from WM-811K, and any automatic action without human approval.

## Data
- **WM-811K** (Kaggle `qingyi/wm811k-wafer-map`, file SHA-256 `1d04fccb…5c65`): 811,457 wafer maps from one fab. 172,950 are labelled by the source fab's engineers; the 638,507 unlabelled ones are not used.
- **Classes are imbalanced:** 85% `none`; Near-full is 149 wafers, Donut 555.
- **Split by lot, not by wafer:** 72/14/14, seed 98, stratified by class. Lots that share an identical map (3,234 maps recorded twice under different lot names) are kept in the same split; one oversized linked group (12,596 wafers) is pinned to train. Details: `docs/results/data_report.md`; split file: `splits/wafer_split_v1.csv.gz`.
- **Label noise:** at least 14 identical maps carry conflicting labels.

## Model
- **Input:** each map resized to 64×64 as area fractions, in 3 channels (wafer, pass, fail fraction).
- **Network:** `small_cnn`, 4 conv blocks of 3×3 conv + BN + ReLU ×2, channels 32→256, global average pool, dropout 0.3, 9 outputs; 1.2 M parameters.
- **Training:** AdamW lr 1e-3, batch 256, 30 epochs with cosine decay. Weighted cross-entropy (√ class weights) plus a fresh random 15,000 `none` wafers per epoch. All 8 rotations/flips as augmentation. No early stopping.
- **Calibration:**
  - The rebalanced training shifts the class priors, so that shift is undone exactly, then one temperature is fitted on validation.
  - A defect-recall operating point (target 0.95) only ever *adds* defect calls to the argmax.
  - The abstention threshold targets 95% selective accuracy on predicted defects.
- **Hardware:** trained on **CPU** (4-core Xeon, PyTorch 2.14) because no GPU credits were available. **Not trained on AMD hardware.**

## Results (validation, 24,067 wafers; test pending)
| Model | Macro-F1 (95% CI by lot) | Accuracy | Defect recall / precision | CPU ms/wafer |
|---|---|---|---|---|
| v0 (1 CNN) | 0.902 (0.885–0.913) | 98.0% | 0.950 / 0.961 | 4.0 |
| v0-tta (1 CNN, 8-way TTA) | 0.911 (0.894–0.923) | 98.1% | 0.950 / 0.963 | 27.3 |
| **v0-ens3 (3 CNNs) — current v1 candidate** | **0.912 (0.896–0.922)** | 98.1% | 0.950 / 0.962 | 9.5 |
| Classical baseline (features + gradient boosting) | 0.865 (0.847–0.878) | — | 0.950 / 0.881 | <1 |

- **Accuracy is not the headline.** "Always none" would already score about 85%. Macro-F1 weights all 9 patterns equally.
- **Per class (v0):** Edge-Ring 0.98, Near-full 0.97, Center 0.95, Donut 0.88, Random 0.88, Edge-Loc 0.87, Scratch 0.80, Loc 0.79, none 0.99.
- **Calibration:** expected calibration error on predicted defects is 0.010; 1.1% of wafers are routed to review.
- **Leakage check:** a wafer-random split scores the same as the lot-grouped one (0.902 vs 0.902), so the score is not inflated by lot leakage.

**Test results: pending** (`docs/results/final/`).

## Explanations
The defect box is chosen by Grad-CAM heat and drawn around the full fail-die cluster. On synthetic patterns pasted onto clean validation wafers, the box overlaps the true defect (IoU ≥ 0.3) for 97% of scratches, 99% of local blobs and 100% of edge arcs (`docs/results/v0_val/localisation.json`). This is an attribution, not a trained detector; there were no box labels.

## Known limitations
- **Main error is missed defects:** 14% of Scratch and 10% of Loc wafers are called `none`. The recall operating point and `needs_review` mitigate this but don't remove it.
- **Small wafers** (≤754 dies) score worse: macro-F1 0.79 versus 0.90 for the largest quartile.
- **Single-label only:** mixed-pattern wafers will be forced into one class. They should be flagged, but this hasn't been measured (MixedWM38 not run).
- **One fab:** performance on other fabs or products is unknown.
- **Small-class intervals:** results for Near-full (20 val wafers) and Donut (82) have wide confidence intervals.

## Reproduce
See `docs/results/OVERNIGHT.md` §6 and `README.md`.
