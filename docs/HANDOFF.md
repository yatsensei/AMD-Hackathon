# Hand-off: data, CNN, factory, and agent

Phases **1–2** (data + CNN), **4** (factory simulation), and **5** (Token Factory agent) are implemented on branch `feature/wafer-agent`. This page is what you can use today, what is still pending, and the rules that keep our numbers honest.

## What works vs pending

| Area | Status |
|---|---|
| Dataset prep + lot-grouped split | Done (`prepare_data.py`, `splits/wafer_split_v1.csv.gz`) |
| CNN classify / calibrate / abstain (`wafer.infer`) | Done (`models/v0`, `v0-ens3`, `v0-tta` in repo; CPU-trained) |
| Factory stream, alerts, root causes, impact | Done (`wafer/factory/`, `streams/stream_val_unscored.csv` committed) |
| **Phase 5 agent** (Token Factory + tools) | **Done** — CLI + `WaferAgent`; tools wrap CNN + factory |
| Classifier registry (`wafer.classifiers`) | Done — CNN default; **experimental YOLO-cls** optional |
| Scored val streams in git | CNN: `stream_val.csv` / `stream_val_ens3.csv`; YOLO: `stream_val_yolo.csv` |
| YOLO real adapter / weights | **Experimental** — `models/yolo-cls-exp` + `wafer/classifiers/yolo.py` (uncalibrated; private split) |
| Dashboard (Phase 6) | Done — agent binds to the stream's `classifier` column |
| Final test-set evaluation | **Pending** — do not use test split for the demo until then |
| CNN retrain on AMD GPU | **Pending** — pitch must not claim AMD training |

## 1. Setup (any laptop; GPU optional)

```bash
git clone -b feature/wafer-agent https://github.com/yatsensei/AMD-Hackathon && cd AMD-Hackathon
pip install -r requirements.txt torch torchvision   # torch needed for CNN scoring / infer
scripts/download_data.sh            # public Kaggle archive, or set KAGGLE_API_TOKEN
python scripts/prepare_data.py      # ~7 min → data/interim/labeled.pkl (+ processed artifacts)
cp .env.example .env                # fill AMD_* for the agent (never commit .env)
```

Split-only (no maps): `splits/wafer_split_v1.csv.gz` is in git.

## 2. Classifier API (agent · dashboard · factory)

```python
from wafer.infer import classify, classify_batch, explain     # $WAFER_MODEL_DIR, else models/v1, else models/v0
r = classify(wafer_map)            # 0 = no die, 1 = pass, 2 = fail
# pattern, confidence, probs, top2, defect_prob, needs_review, review_reasons, model_version
```

Prefer the pluggable backend (Phase 5 / future YOLO):

```python
from wafer.classifiers import get_classifier
clf = get_classifier()                 # cnn by default ($WAFER_CLASSIFIER)
r = clf.classify(wafer_map)            # also sets classifier="cnn"
# Experimental YOLO-cls (pip install ultralytics; models/yolo-cls-exp/best.pt):
# clf = get_classifier("yolo")         # uncalibrated; needs_review always False
# Missing ultralytics/weights → ClassifierNotAvailableError (never silent CNN fallback)
```

- ~4–10 ms/wafer on CPU (`models/v0` ~4 ms; `models/v0-ens3` ~9.5 ms).
- CNN calibration / `needs_review` rules apply to CNN only — **never** reuse τ/θ for YOLO.
- YOLO confidence is raw softmax. It was **not** trained on `wafer_split_v1` and overlaps the val demo stream — do not claim fair comparison. See `models/yolo-cls-exp/README.md`.
- YOLO demo stream: `python -m wafer.agent --score-stream streams/stream_val_unscored.csv --classifier yolo -o streams/stream_val_yolo.csv`
- When `needs_review` is True, say the model is unsure and cite `review_reasons`. Never call 0.4 “likely”.

**Dashboard (Phase 6):** `from wafer.viz import render_wafer, render_heatmap`. Grad-CAM box = attribution, not detection.

## 3. Factory stream (Phase 4)

```bash
python scripts/make_stream.py                 # val + CNN → streams/stream_val.csv (+ alerts)
python scripts/make_stream.py --no-predict    # → streams/stream_val_unscored.csv (committed)
```

```python
from wafer.factory import (
    build_stream, add_predictions, score_stream, check_alert, check_lot_history,
    lots_caught_early, lookup_root_cause, estimate_impact, impact_from_alert,
)
```

- Use **val** while building; switch to **test** only after the final test evaluation.
- Alert logic reads model `pattern`, never true `label`.
- Scenario: 20×25 wafers; from LOT-010, ETCH-04 Edge-Ring excursion.

## 3b. Agent (Phase 5) — how to run

**Dependencies:** `openai`, `python-dotenv` (in `requirements.txt`). CNN scoring also needs `torch`.

**Env** (copy `.env.example` → `.env`; never commit secrets):

| Variable | Purpose |
|---|---|
| `AMD_API_KEY` | Token Factory key (`rc-…`). Optional alias: `RADEON_API_KEY` |
| `AMD_BASE_URL` | e.g. `https://developer.amd.com.cn/radeon/api/v1` |
| `AMD_MODEL` | Id from `GET $AMD_BASE_URL/models` — catalog changes; do not hard-code |
| `WAFER_CLASSIFIER` | `cnn` (default). `yolo` → clear unavailable error |
| `WAFER_MODEL_DIR` | Optional CNN export (else `models/v1`, then `models/v0`) |

```bash
# Score unscored stream (needs labeled.pkl + a CNN export)
python -m wafer.agent --score-stream streams/stream_val_unscored.csv -o streams/stream_val.csv

# Ask — stdout is the answer; errors/evidence warnings on stderr; timings only with --debug
python -m wafer.agent -q "What is happening on ETCH-04, and what exposure has been observed?" \
  --stream streams/stream_val.csv --as-of "2026-10-13 04:10"

python -m wafer.agent --help          # full CLI
python -m wafer.agent -q "..." --stream streams/stream_val.csv --debug   # timings + footer
```

```python
from wafer.agent import WaferAgent
import pandas as pd
df = pd.read_csv("streams/stream_val.csv")
agent = WaferAgent.from_env(df, as_of="2026-10-13 04:10")
print(agent.ask("Summarise the excursion and observed exposure.")["answer"])
```

**Tools:** `classify_wafer`, `lookup_root_cause`, `check_lot_history`, `estimate_impact`.

**Evidence rules (implemented):**
- History is compact for the LLM (counts, lot summaries, alerts, review candidates, event sample); full events stay local on the session.
- Alerts use full history ≤ `as_of` before sampling; lot filters do not hide tool-wide alerts. No predictions → alerts unavailable (never use labels as predictions).
- Empty alerts ≠ “no exposure / no action” — only that the rule has not fired on available predictions.
- Impact: **subsequent lots observed** since the alert; `observed_wafers` vs `assumed_full_lot_*`; never claim unobserved wafers were processed; never claim savings unless `intervention_recorded`.
- Root causes: possibilities + `source` + `source_warnings`.
- Scored rows record `classifier` + `model_version`.

**AMD story:** CNN trained on **CPU**. Live AMD piece = agent LLM on Token Factory.

## 4. YOLO benchmark (Phase 3) — pending adapter

- PNGs: `python scripts/prepare_data.py --yolo` → `yolo_data/...`
- Score like the CNN: predictions CSV → `python -m wafer.evaluate --predictions ...`
- Agent/dashboard integration: implement `wafer/classifiers/yolo.py` for real (replace the stub). Do not invent boxes/outputs beforehand.

## 5. Numbers you may quote
- **Validation:** `docs/results/OVERNIGHT.md`, `docs/results/v0_val/` (slightly optimistic).
- **Pitch / test numbers:** only from `docs/results/final/` after `--final` (not yet).
