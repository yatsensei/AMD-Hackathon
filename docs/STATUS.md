# Status board

One line per item from spec Sections 3.10, 6.1–6.3, 7.6 and 9.1–9.4, plus Phases 4–5. ✅ done · 🟡 in progress / partly done · ❌ not started. Update it in every PR.

## Phase 1 — data (Nasser & Jina)
| Item | Owner | Status | Notes |
|---|---|---|---|
| `scripts/download_data.sh` | Nasser / Jina | ✅ | Works with `KAGGLE_API_TOKEN`, legacy key, or no credentials (public archive) |
| `wafer/data/load.py` + pickle compat patch | Nasser / Jina | ✅ | Real file needs the `pandas.indexes` alias **and** latin1 (Python 2 pickle); logged |
| Clean (3.4) | Nasser / Jina | ✅ | 811,457 rows → 172,950 labelled (0 invalid, 0 empty) |
| Encodings A + B, scratch-survival tests (3.5) | Nasser / Jina | ✅ | |
| Duplicates → lot linking (3.6) | Nasser / Jina | ✅ | 3,250 duplicate groups, 14 with conflicting labels; giant linked group pinned to train |
| Grouped split with seed search (3.7) | Nasser / Jina | ✅ | Seed 98, 72/14/14, worst class-share deviation 7.5%; Near-full in 137 lots |
| **`wafers_s64.npz` + `metadata.parquet`** | Nasser / Jina | ✅ | Built in Claude's environment (7 min). Not in git: rebuild with `scripts/prepare_data.py` |
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
| `metrics.py`, `evaluate.py` (val; `--final` once; `--predictions` CSV) | Anisha / Akein | ✅ | |
| `wafer/infer.py` + `wafer/explain.py` + `wafer/export.py` | Anisha / Akein | ✅ | Serving `models/v0` (CPU); sanity inputs pass |
| E0 baseline → `models/v0` | Anisha / Akein | ✅ (CPU) | Val macro-F1 0.902; `v0-tta` 0.911; `v0-ens3` 0.912. **CPU-trained**. See `OVERNIGHT.md` |
| E0b classical baseline | Claude | ✅ | Val macro-F1 0.865; CNN significantly better (+0.037) |
| Tier 1 sweep + E-LEAK + E-COMB | Anisha / Akein | 🟡 | Single-seed CPU screening done. Needs 5 seeds on an AMD GPU |
| Freeze `models/v1` | Akein | 🟡 | Recommended: `models/v0-ens3`; waiting for the team's decision |
| Final test evaluation, once | Anisha / Akein | ❌ | |
| Tier 2 (E9, E10, E11, E8, E4, E7) | Anisha / Akein | ❌ | Low priority |
| Explainability check (spec 8.4) | Claude | ✅ | Box IoU≥0.3 on synthetic; `docs/results/v0_val/gradcam_examples.png` |
| Teammate hand-off (`docs/HANDOFF.md`, split file, data access) | Claude | ✅ | Updated for Phase 5 |
| Model card | Claude | 🟡 | Draft; test numbers pending the final run |
| Accuracy improvements (ResNet-18, 60 epochs, bigger ensemble) | Claude | ✅ | None beat `models/v0-ens3` (0.912) |

## Phase 3 — YOLO (experimental)
| Item | Owner | Status | Notes |
|---|---|---|---|
| YOLO-cls weights + `classifiers/yolo.py` adapter | teammate / integration | 🟡 experimental | `models/yolo-cls-exp`; Ultralytics classify; uncalibrated; private stratified split ≠ `wafer_split_v1`; overlaps val demo — not a fair CNN comparison |
| Separate YOLO val-demo stream | integration | ✅ | `streams/stream_val_yolo.csv` (`classifier=yolo`, `model_version=yolo-cls-exp`) |
| Retrain on shared lot-grouped split | TBD | ❌ | Required before any honest compare / combine rule |

## Phase 4 — factory simulation (Nasser & Shefali Cat)
| Item | Owner | Status | Notes |
|---|---|---|---|
| `wafer/factory/root_causes.json` (9 patterns) | Nasser / Shefali Cat | 🟡 | 7 of 8 defect patterns cited (arXiv 2607.00961 Table 1, US patents, Minitab), all from search excerpts: open each link to confirm, then drop "Not read in full". Near-full: no source found |
| `build_stream` + `scripts/make_stream.py` | Nasser / Shefali Cat | ✅ | Val ETCH-04 Edge-Ring from LOT-010; `stream_val_unscored.csv` committed |
| `check_alert`, `lots_caught_early`, `estimate_impact` | Nasser / Shefali Cat | ✅ | Observed vs assumed-full-lot exposure; no false savings claims; one alert per tool, later patterns in `also_seen` |
| `check_lot_history` + `score_stream` | Phase 5 | ✅ | Compact LLM payload; full events local; as_of-safe alerts |
| Scored stream with CNN predictions | Nasser | ✅ | `streams/stream_val.csv` (models/v0, built on Kaggle). 264/275 excursion wafers called Edge-Ring; alert ETCH-04/Edge-Ring at LOT-010, none on lots 1–9. Extra alert ETCH-04/Edge-Loc at LOT-013 = CNN misreads of Edge-Ring wafers |
| Test-split stream | Nasser | 🟡 | `streams/stream_test_unscored.csv` committed (500 test wafers, no overlap with val stream). Score it only after the final test evaluation: `python scripts/make_stream.py --split test` |

## Phase 5 — agent
| Item | Owner | Status | Notes |
|---|---|---|---|
| `wafer/classifiers/` (CNN + experimental YOLO) | Phase 5 | ✅ | Default CNN; YOLO optional; no silent fallback |
| AMD Token Factory client + tool loop | Phase 5 | ✅ | Tools: classify_wafer, lookup_root_cause, check_lot_history, estimate_impact |
| CLI + callable `WaferAgent` | Phase 5 | ✅ | `python -m wafer.agent`; `--debug` for timings; answer-only by default |
| Parallel tool calls + compact history | Phase 5 | ✅ | Status flow: 1× history then parallel root-cause + impact |
| Tests (`test_classifiers`, `test_agent`, factory) | Phase 5 | ✅ | Mocked LLM; no live key required |
| Docs (README, HANDOFF, STATUS, AMD_SETUP, `.env.example`) | Phase 5 | ✅ | How to run + works vs pending |
| Wider questions: 7 more tools (`wafer/factory/insights.py`) | Agent v2 | ✅ | line_overview, defect_trend, find_wafers, explain_wafer, describe_pattern, model_performance, get_decisions |
| Conversation memory + line snapshot + UI context | Agent v2 | ✅ | Last 6 Q/A pairs per chat; snapshot of ids/alerts/decisions in the system message; dashboard sends page / lot / wafer |
| Live evaluation | Agent v2 | ⏳ | `python scripts/agent_eval.py` (needs AMD key) writes `docs/results/agent_eval.md`; not run yet |

## Phase 6–7 — pending
| Item | Status | Notes |
|---|---|---|
| Dashboard (`dashboard/`, FastAPI + static UI) | ✅ | Overview, Alerts, Tools, Lots, Review queue, Decisions + copilot chat bubble; see `dashboard/README.md` |
| YOLO second opinion in the dashboard | 🟡 | Available when ultralytics + weights present; agent follows stream `classifier` |
| End-to-end demo polish | ❌ | stream → CNN → alert → agent → UI |
