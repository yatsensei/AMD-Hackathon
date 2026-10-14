# Design review: Phase 1 & 2 spec (v1 → v2)

**Date:** Wed 14 Oct 2026 · **Reviewed:** `docs/specs/SPEC_phase1_phase2.md` as first written (the v1 commit)
**Outcome:** v1 rewritten as v2 in place. This file explains every change. The spec is still the source of truth; this is the reasoning behind it.

We went through every decision in v1 and asked three things: *is it correct*, *is it the best use of four days*, and *will it hold up when a judge pushes on it*. Most of v1 holds up. Lot-grouped splits, a locked test set, macro-F1 over accuracy, area-fraction encoding and abstention are all right, and they stay. The changes below fix things that would have **produced a wrong or indefensible number**, **broken the latency contract**, or **wasted the critical path**.

Evidence scripts are in `docs/specs/evidence/`. All numbers below come from running them on a 4-thread CPU.

---

## 1. Ten changes that matter most

| # | v1 said | Problem | v2 says |
|---|---|---|---|
| 1 | Final test run "once **per candidate model**" | Scoring several candidates on test and then picking one is model selection on test. Rule 0.1 contradicted itself | Test is scored **once**, for **one** frozen model. Comparators (E0 baseline, classical baseline, YOLO) are named in advance and scored in the same run, never chosen between |
| 2 | Also report on the dataset's original `trianTestLabel == "Test"` subset "for reference" | Our training split contains about 70% of those wafers. The number would be contaminated and meaningless | Removed. Replaced with **E-LEAK**: the same config trained on a wafer-random split vs a lot-grouped split (inside train+val only). The gap measures the leakage most published results have. It makes our honesty claim measurable, so it's our best pitch slide |
| 3 | `resnet18s` (3×3 stride-1 stem, no maxpool); TTA ×8; ensemble of 3–5; CPU < 50 ms | Measured: **2.2 GMACs, 20 ms per wafer** for one forward pass. With D4 TTA it's **101 ms**, and with 5 members × TTA it's **~670 ms**. That's 13× over budget. v1's own plan breaks v1's own contract | **Latency is a hard gate for v1 eligibility.** The served config (members × TTA) must meet the budget on the reference CPU. `resnet18s` uses a stride-2 stem (0.56 GMACs, 6.7 ms). The stride-1 stem is allowed only without TTA. New Rule 0.10: *the model we report is the model we ship* |
| 4 | Grad-CAM on the last conv layer | `small_cnn`'s last map is **4×4** and `resnet18s`'s is 8×8. A 4×4 heatmap can't draw a box around a one-die-wide scratch | CAM comes from a layer that is **≥ 16×16**. `small_cnn` drops its last max-pool (same FLOPs, 8×8 output). Localisation is checked with a **synthetic-insertion benchmark** (paste a known pattern onto a clean wafer, then measure pointing-game and box IoU), so "the box is right" is measured, not just shown |
| 5 | Temperature scaling = calibration | Training rebalances the classes (downsampling + class weights), so the logits assume the wrong class priors. **Temperature scaling can't change the argmax, so it can't fix a prior shift.** "Calibrated" probabilities would still over-predict defects on a real fab's mix | First **correct for the prior shift** (add `log π_natural − log π_train_effective` to the logits, worked out exactly, no fitting), then fit T. One extra line of code, and the probabilities then mean what they say |
| 6 | Pattern = argmax; separate defect-recall target | Nothing actually sets the operating point, so defect recall is whatever comes out of training | **Two-stage decision:** call it `none` only if `p(none) ≥ θ_none`, with θ chosen on val to hit the defect-recall target; otherwise take the argmax over defect classes. One threshold, estimated on about 3.8k val defects, and it maps to the fab's real cost: a missed defect costs more than a false alarm |
| 7 | ECE ≤ 0.05 and selective accuracy ≥ 0.97 at 90% coverage as targets | 85% of wafers are easy `none` at p≈0.99. Both targets **pass even if the model is useless on defects** | Targets and abstention threshold are defined on **predicted-defect wafers**, with per-class review rates reported. We also report classwise ECE. Overall ECE is still reported, but not as a target |
| 8 | Early stopping on val macro-F1 (patience 7), with a cosine schedule | (a) Early stopping cuts a cosine schedule off while the learning rate is still high, so the two don't fit together. (b) Val macro-F1 is noisy: about 22 val Near-full wafers, so two wafers move macro-F1 by about 0.01. Stopping on it is another hidden tuning knob using val | **Fixed-length cosine schedule.** Use the final checkpoint. Epoch count is chosen once from E0's curves. Val metrics are logged every epoch for diagnosis only |
| 9 | McNemar test between final and E0 | McNemar treats wafers as independent, but wafers in a lot are correlated, so p-values come out too small. It also tests **accuracy**, which is mostly `none`, not our headline metric | **Paired lot-bootstrap** of Δmacro-F1 and Δdefect-recall (95% CI excludes 0 ⇒ significant). McNemar is still reported, with a caveat |
| 10 | Tier 1 is sequential, "one thing at a time from the current best" | Each step waits for a person to read the last result. That's about 6 serial human round-trips on Friday, the day before freeze | **Parallel one-factor-at-a-time sweep** around E0 (all launched Friday morning, many processes on one MI300X), then **one combination run** with the winners, confirmed against E0 over 5 seeds |

---

## 2. Decision-by-decision review

Verdicts: **Keep** · **Change** · **Drop** · **Add**.

### Rules (Section 0)
| Decision | Verdict | Reasoning |
|---|---|---|
| Locked test split | **Change** | Kept, but tightened to one model and pre-registered comparators (change #1). Also: Phase 4's demo stream is built from test, so the dashboard team would see test predictions every day from Thursday on, and that's peeking. v2: the stream is built from **val** until the final evaluation has run, then rebuilt from test |
| Split by lot | **Keep** | Correct and central to the pitch. We add a stricter stress test (E11, unseen wafer geometry) because lot-grouping doesn't stop *product*-level shortcuts |
| Rebalance only in training | **Keep** | But rebalancing creates a prior shift that must be undone at inference (change #5) |
| Macro-F1 primary | **Keep + add** | We also report **8-class defect macro-F1**. `none` F1 is about 0.99 and pads the 9-class average |
| One class list | **Keep** | |
| No data/credentials in git | **Keep + add** | `LSWMD.pkl` is a pickle, which means loading it can run arbitrary code. Check its SHA-256 before unpickling. Ship weights as `safetensors`, or load with `torch.load(weights_only=True)` |
| Reproducible from config | **Keep** | |
| Save GPU credits | **Keep, and rethink what the GPU is for** | The 64×64 labelled set is about 2 GB as uint8 and fits on a 192 GB MI300X many times over, so compute isn't the bottleneck; people's time is. Use the GPU for things a laptop can't do: many seeds in parallel, E9 on all 811k wafers, ensembles. That also makes "trained on MI300X" a real claim rather than decoration |
| *(new)* 0.10 "Report what you ship" | **Add** | Change #3 |

### Data (Sections 2–3)
| Decision | Verdict | Reasoning |
|---|---|---|
| Kaggle source, counts | **Keep** | The counts add up exactly (25,519 defects + 147,431 `none` = 172,950) |
| Pickle compat patch | **Keep** | Convert once, then never load the raw pickle again |
| Python 3.11 pinned | **Change** | The ROCm PyTorch images ship their own Python. Pinning 3.11 against them wastes an hour. v2: use the image's Python on the GPU box and ≥3.10 elsewhere; `.npz`/`.parquet` work across versions |
| Encoding A (nearest) as baseline | **Change** | **Evidence:** a 1-die diagonal scratch on a 200×200 wafer keeps **45/140** pixels under nearest-neighbour. Move it **one die sideways** and it keeps **0**, so the scratch is gone entirely. Area-fraction keeps 126 pixels both times. B becomes the default from E0; A is kept only as the E1 comparator |
| Encoding B spec text | **Change** | The v1 channel description contradicted itself (`fail_mask * wafer_mask_inverse_safe`). Also, INTER_AREA vs NEAREST "depending on size" is unnecessary, because INTER_AREA on float masks handles both directions. v2 has one clean definition |
| *(new)* Radius channel | **Add (E1c)** | Center, Donut, Edge-Ring and Edge-Loc are mostly *radial* ideas. A CNN with global pooling has to work out position from the wafer edge. A normalised-radius channel (computed from the wafer mask) gives it position directly, costs nothing, and doesn't change under D4 |
| Exact-hash dedupe; drop cross-split dupes from val/test | **Change** | Two *all-pass* wafers of the same shape are identical maps but different physical wafers. Dropping them takes easy `none` wafers out of val/test and skews the numbers. v2: drop a cross-split duplicate only if it has **≥ 1 fail die**, because an identical fail pattern means the same wafer recorded twice. Duplicates with **conflicting labels** are counted and reported as a measured **label-noise floor** (useful for the pitch) |
| `StratifiedGroupKFold(20)`, 3/3/14 | **Keep + change** | Runtime is fine (**~6 s** at full size). But in a worst-case simulation where a rare class sits in few lots, Near-full landed **50 in test, 0 in val**. Changing `random_state` by hand until it passes is slow and unlogged. v2: **search seeds 0–99 automatically**, keep the split with the lowest worst-case relative class deviation that meets minimum per-class counts, and log the seed. Phase 1 must report **lots per class**, because it decides how wide the per-class CIs are |
| ±2 pp proportion assertion | **Change** | Absolute pp is meaningless for 0.09% classes and loose for 85% ones. v2 uses a relative tolerance plus minimum counts |
| `cv_fold` "(0–13)" with `n_splits=5`; CV "on train+val" but folds only on train | **Fix** | Two bugs in the spec. v2: `cv_fold ∈ 0–4`, assigned over **train+val** lots |
| YOLO PNGs on the Phase 1 critical path | **Change** | Phase 2 doesn't need 172k PNGs. They move off the critical path (Thursday afternoon). Phase 1's critical path becomes: clean → split → dedupe → npz + parquet |
| 64×64 default, 96 stretch | **Keep** | |

### Training (Sections 4–5)
| Decision | Verdict | Reasoning |
|---|---|---|
| DataLoader with 4–8 workers | **Change** | Not needed. Load the whole npz **onto the GPU** once and index tensors directly, with augmentation in-batch on GPU. This removes the "workers misbehave" risk and is faster |
| ROCm notes | **Add** | `MIOPEN_FIND_MODE=FAST` (otherwise MIOpen's kernel search can make the first epochs very slow), plus a writable `MIOPEN_USER_DB_PATH` inside Docker. `torch.compile` is optional; turn it off at the first error |
| Per-epoch `none` downsampling + √ class weights | **Keep** | Plus prior correction at inference (change #5) |
| D4 augmentation | **Keep** | Exact on a square grid. Correct for this domain |
| Small rotation ±15° | **Keep (ablate)** | |
| Die-flip noise | **Change** | v1 doesn't say what "flip a die" means for fractional encoding B. v2 defines it per encoding |
| No crops/translations | **Keep** | |
| `small_cnn` | **Change** | Drop the 4th max-pool. FLOPs stay the same and the final map goes from 4×4 to 8×8 (change #4) |
| `resnet18s` stride-1 stem | **Change** | Stride-2 stem by default (change #3). Measured: 0.56 vs 2.2 GMACs; 6.7 vs 20 ms |
| `resnet18s_pt` (ImageNet init) | **Drop** | ImageNet features on 3-channel categorical maps are a weak prior, and the conv1 weight surgery is fiddly. Domain pretraining (E9) is the version of this question worth answering |
| timm backbones (E7) | **Demote** | Only if the GPU is idle and someone is free. They won't change the story |
| *(new)* Classical baseline E0b | **Add** | Hand-crafted features (radial-band densities, region densities, Radon/geometry, fail fraction) + `HistGradientBoostingClassifier`, on CPU, in minutes. This is the standard WM-811K approach from the literature. If the CNN doesn't clearly beat it, we need to know before the pitch. It's also a more honest "second model family" than YOLO-cls trained on PNGs of the same 64×64 maps |
| AdamW / cosine / bs 256 / bf16 / clip | **Keep** | |
| Early stopping | **Drop** | Change #8 |
| 3 seeds | **Change → 5** | Runs take minutes and fit side by side on one MI300X. Five seeds make the "within one SD" selection rule mean something |
| Shared `leaderboard.csv` appended by every run | **Change** | Parallel runs and parallel PRs cause merge conflicts. Each run writes `experiments/<id>/summary.json`; `scripts/aggregate.py` rebuilds the CSV |
| `env.json` with git commit | **Keep + add** | Record whether the working tree was dirty. A run from a dirty tree can't be reproduced |

### Experiments (Section 6)
| Decision | Verdict | Reasoning |
|---|---|---|
| E0 → hand off fast | **Keep** | |
| Sequential Tier 1 | **Change** | Change #10 |
| E2 mixes loss and sampler in four options | **Keep, relabelled** | Fine as a sweep, but report it as "imbalance recipe", not as a single factor |
| E5 TTA, E6 ensemble | **Keep, gated** | Only in a form that meets the latency budget |
| *(new)* E-LEAK | **Add, Tier 1** | Change #2. One extra config, and the most quotable number we'll have |
| E8 aux scalars | **Keep + warning** | `dieSize` and wafer shape are stand-ins for *product*. The model can learn "this product tends to get Edge-Ring", a shortcut that helps on this fab and fails on any other. Check against E11 |
| E9 unlabelled data | **Promote to first Tier 2 item; pseudo-labelling first** | Using all 811k wafers is our most distinctive technical story, and it's work that genuinely needs the GPU. Pseudo-labelling with a teacher ensemble reuses existing code. SimCLR only if there are hours to spare. Keep v1's lot-exclusion rule |
| E10 MixedWM38 | **Change** | MixedWM38 comes from a different source, so low confidence there could mean *domain shift*, not "it recognised a mixed pattern". v2 compares abstention on its **single-pattern** wafers (the control) vs its **mixed** wafers. Only the *difference* supports the claim |
| *(new)* E11 unseen geometry | **Add, Tier 2** | Hold out whole wafer-shape groups (`orig_h, orig_w`) in CV on train+val. This approximates "a new product", the real deployment question |
| Optional 5-fold CV | **Keep** | With the fold bug fixed |

### Evaluation & calibration (Section 7)
| Decision | Verdict | Reasoning |
|---|---|---|
| Metric list | **Keep + add** | Add 8-class defect macro-F1, classwise ECE, AURC, per-class review rate |
| Slices | **Keep** | Good list |
| Bootstrap by lot | **Keep** | Correct. Near-full CIs will be wide; say so |
| McNemar | **Demote** | Change #9 |
| Temperature fit; "fit T on averaged logits" (§7.4) vs "average calibrated probabilities" (E6) | **Fix** | The two sections contradicted each other. v2: average member **probabilities**, take the log, apply the prior correction, then fit one T on the ensemble |
| τ from selective accuracy ≥ 0.98 | **Change** | Change #7 |
| Confusable-pair margin 0.15 | **Keep, fixed** | Kept as a fixed constant rather than tuned, to avoid another threshold fitted on val |
| Error gallery | **Keep + add** | Optional blind relabel: two people label the 50 most-confident val errors without seeing labels. That gives a measured label-noise estimate for the pitch |
| Calibrating and tuning on the same val | **Accept, disclose** | 1–2 parameters on about 26k wafers is low risk. Post-calibration val numbers are labelled optimistic; test is the honest number |

### Explainability (Section 8)
| Decision | Verdict | Reasoning |
|---|---|---|
| Grad-CAM + box instead of detection | **Keep the idea, change the layer** | Change #4. Call it **attribution**, not detection, in the pitch. A judge who knows the difference will ask |
| Box = CAM ∩ fail dies, largest component | **Keep + morphology** | Add a small closing step so a dotted scratch becomes one component |
| *(new)* Synthetic-insertion benchmark | **Add** | Change #4 |

### Interfaces (Section 9)
| Decision | Verdict | Reasoning |
|---|---|---|
| `model = load_model(...)` then module-level `classify(m)` with no model argument | **Fix** | Ambiguous: which model does `classify` use? v2: `load_model` returns a `WaferClassifier` with `.classify/.classify_batch/.explain`. The module-level `classify/explain/classify_batch` stay, using a lazily loaded default (`$WAFER_MODEL_DIR` or `models/v1`), **so Phase 5/6 call sites don't change** |
| Output dict | **Extend (no breaking changes)** | Add `defect_prob` (= 1 − p(none)) and `review_reasons`. Every value is a plain Python type, so the agent tool can JSON-serialise it directly |
| *(new)* Train/serve parity test | **Add** | For 200 val uids, `classify(raw_map)` must match the eval pipeline's probabilities to 1e-5. This catches the most common way a good model shows bad numbers in a demo |
| Latency test "< 50 ms" with no hardware stated | **Fix** | Reference: 4 threads (`torch.set_num_threads(4)`), batch 1, served config, measured with `scripts/bench_latency.py`. The pytest is a looser smoke test (< 150 ms) so it doesn't flake on laptops |
| Release ≤ 50 MB in git | **Keep** | Weights as `safetensors` |

---

## 3. Notes for other phases (not ours to change, raised here for the team)
- **Phase 3 (YOLO):** YOLO-cls on PNGs of the same 64×64 maps is a second CNN, not a different model family, and nearest-rendered PNGs lose scratches (see the evidence above). Consider rendering from encoding B and putting the effort into E0b or the synthetic-localisation benchmark instead. Whatever it is, it's scored through `evaluate.py --predictions`, as one of the pre-registered comparators.
- **Phase 4:** build the demo stream from **val** until the final evaluation; swap to **test** after Saturday 10:00.
- **Phase 5:** when `needs_review` is true, say why using `review_reasons`. The agent should never paraphrase a 0.4 confidence as "likely".
- **AMD story:** if the agent's LLM is also served on the MI300X (vLLM on ROCm), the "end to end on AMD" claim gets much stronger than training a 1M-parameter CNN on it.

## 4. What we deliberately did *not* change
- **Single-label classification.** Mixed patterns are out of scope and should be flagged. Going multi-label would mean new data (MixedWM38) and a new head, too much for this timeline.
- **64×64 resolution.** It's the literature standard and fits the CPU budget. 96 stays as a stretch goal.
- **Sticking with WM-811K.** It's the only large real-fab public set. Its label noise is a fact we report, not a reason to switch.
