# Data report — WM-811K (Phase 1)

Generated Wed 14 Oct 2026 13:18 UTC · git `37098bc (dirty)`
Command: `python scripts/prepare_data.py --jobs 4`
Versions: python 3.13.16 · numpy 2.5.4 · pandas 3.0.6 · opencv 5.0.0 · scikit-learn 1.9.1

## File
- **SHA-256 of `LSWMD.pkl`:** `1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65`
- **First recording** — no hash was on record, so the file was unpickled unverified. Commit this report so everyone else is checked against it.

## Counts
| | wafers |
|---|---|
| raw rows | 811,457 |
| labelled | 172,950 |
| unlabelled (metadata kept for E9) | 638,507 |
| dropped: not 2D / values outside {0,1,2} | 0  |
| dropped: no dies at all | 0 |
| **final labelled wafers (npz + metadata)** | **172,950** |
| missing `lotName` filled as singleton lots | 0 |
| missing `waferIndex` (set to -1) | 0 |

Encodings written for sizes: `wafers_s64.npz`.

## Split (lot-grouped, stratified, 70/15/15)
- Actual shares: train 72.2% · val 13.9% · test 13.9%.
- Seed search over 100 seeds: **100** met the minimum counts. Chosen seed **98**, worst relative class-share deviation in val/test **0.075** (Edge-Ring in val).
- Minimum per-class counts in val and test: {'Center': 100, 'Donut': 50, 'Edge-Loc': 100, 'Edge-Ring': 100, 'Loc': 100, 'Near-full': 15, 'Random': 100, 'Scratch': 100, 'none': 100}. Maximum allowed relative deviation: 0.25.
- Split unit: **split group** = a lot, or several lots linked because they share a duplicate map (see Duplicates). 10,129 groups.
- Lots per class decide how wide the per-class confidence intervals are. Bootstrap by `split_group`.

| class | train wafers | val wafers | test wafers | total wafers | train lots | val lots | test lots | total lots | share % |
|---|---|---|---|---|---|---|---|---|---|
| Center | 3,024 | 635 | 635 | 4,294 | 1,164 | 248 | 236 | 1,648 | 2.483 |
| Donut | 391 | 82 | 82 | 555 | 166 | 32 | 35 | 233 | 0.321 |
| Edge-Loc | 3,700 | 745 | 744 | 5,189 | 2,312 | 434 | 465 | 3,211 | 3.000 |
| Edge-Ring | 6,784 | 1,448 | 1,448 | 9,680 | 757 | 168 | 150 | 1,075 | 5.597 |
| Loc | 2,584 | 505 | 504 | 3,593 | 1,767 | 354 | 351 | 2,472 | 2.077 |
| Near-full | 108 | 20 | 21 | 149 | 99 | 18 | 20 | 137 | 0.086 |
| Random | 625 | 121 | 120 | 866 | 320 | 70 | 59 | 449 | 0.501 |
| Scratch | 856 | 169 | 168 | 1,193 | 766 | 152 | 141 | 1,059 | 0.690 |
| none | 106,712 | 20,342 | 20,377 | 147,431 | 4,790 | 900 | 901 | 6,591 | 85.245 |
| **all** | 124,784 | 24,067 | 24,099 | 172,950 | 7,721 | 1,522 | 1,519 | 10,762 | 100.000 |

Relative class-share deviation per split (|share in split − overall share| / overall share):

| class | train rel. dev. | val rel. dev. | test rel. dev. |
|---|---|---|---|
| Center | 0.024 | 0.063 | 0.061 |
| Donut | 0.024 | 0.062 | 0.060 |
| Edge-Loc | 0.012 | 0.032 | 0.029 |
| Edge-Ring | 0.029 | 0.075 | 0.074 |
| Loc | 0.003 | 0.010 | 0.007 |
| Near-full | 0.005 | 0.035 | 0.011 |
| Random | 0.000 | 0.004 | 0.006 |
| Scratch | 0.006 | 0.018 | 0.011 |
| none | 0.003 | 0.008 | 0.008 |

### CV folds (train + val lots, seed 0)
| class | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| Center | 732 | 732 | 732 | 731 | 732 |
| Donut | 95 | 94 | 95 | 94 | 95 |
| Edge-Loc | 889 | 889 | 889 | 889 | 889 |
| Edge-Ring | 1,647 | 1,646 | 1,647 | 1,646 | 1,646 |
| Loc | 618 | 618 | 618 | 617 | 618 |
| Near-full | 25 | 26 | 26 | 24 | 27 |
| Random | 150 | 149 | 149 | 149 | 149 |
| Scratch | 205 | 205 | 205 | 205 | 205 |
| none | 25,402 | 25,410 | 25,402 | 25,438 | 25,402 |

## Duplicates (exact hash of the original map)
- Duplicate groups: **3,250** covering 6,502 wafers; 3,248 groups are *informative* (min(fail dies, pass dies) ≥ 1). Uniform maps (all-pass, all-fail) are expected to repeat between different wafers and are ignored.
- **Groups with conflicting labels: 14** (28 wafers). Identical maps with different labels are a measured lower bound on label noise.
- Class mix of duplicated wafers: {'none': 6304, 'Edge-Loc': 67, 'Loc': 39, 'Random': 28, 'Scratch': 22, 'Near-full': 15, 'Edge-Ring': 14, 'Center': 13}
- **Leakage handling:** 3,234 informative maps appear in more than one lot. Their 641 lots were linked into shared split groups, so every copy lands in the same split and no wafer was dropped. Largest group: 12,596 wafers / 598 lots. Verified after splitting: no informative map appears in two splits.
- **Pinned to train:** 12,596 wafers in linked groups larger than 1% of the data. A group that big would otherwise dominate a 15% val or test split and skew its product mix.

## Map sizes (final labelled set)
- Distinct shapes: 346. `orig_h*orig_w`: min 45 · q25 676 · median 1,020 · q75 1,596 · max 28,611.

| orig_h*orig_w bin | wafers |
|---|---|
| (44.999, 676.0] | 48,015 |
| (676.0, 1020.0] | 38,592 |
| (1020.0, 1596.0] | 43,505 |
| (1596.0, 28611.0] | 42,838 |

Most common shapes:

| orig_h | orig_w | wafers |
|---|---|---|
| 25 | 27 | 18,781 |
| 26 | 26 | 14,366 |
| 30 | 34 | 12,400 |
| 29 | 26 | 11,751 |
| 27 | 25 | 10,682 |
| 39 | 37 | 10,173 |
| 33 | 29 | 8,125 |
| 42 | 44 | 6,724 |
| 44 | 41 | 6,134 |
| 41 | 33 | 5,091 |

## Original `trianTestLabel` tag vs our split (reference only — never used for evaluation)
| orig_split | train | val | test |
|---|---|---|---|
| Test | 85,431 | 16,729 | 16,435 |
| Training | 39,353 | 7,338 | 7,664 |
