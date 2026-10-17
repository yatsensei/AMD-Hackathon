# Sentry dashboard

FastAPI backend plus a static single-page UI (no build step). It replays the scored
validation stream as a live etch line, so every screen shows real WM-811K wafers.

```bash
pip install -r requirements.txt -r dashboard/requirements.txt
uvicorn dashboard.server:app --port 8000        # from the repo root
# open http://localhost:8000   (homepage; Sign in leads to the dashboard at /app/, add ?t=400 to start after the excursion)
```

Works without the 2 GB dataset: the 500 stream maps ship in `dashboard/data/stream_maps.npz`
(also used by `classify_wafer` via `wafer.data.access.get_wafer_map`).
The copilot needs `AMD_API_KEY`, `AMD_BASE_URL` and `AMD_MODEL` (see `.env.example`);
everything else runs offline.

**Hugging Face Docker Space:** see [`docs/DEPLOY_SPACE.md`](../docs/DEPLOY_SPACE.md)
(`uvicorn` on `0.0.0.0:7860`).

## Pages

| URL | Page | Files |
|---|---|---|
| `/` | Product homepage | `dashboard/site/index.html`, `site.css`, `site.js` |
| `/signin` | Demo sign-in. No accounts: it only keeps a display name in the browser tab, then opens `/app/` | `dashboard/site/signin.html` |
| `/app/` | The app: `#overview`, `#alerts`, `#tools`, `#lots`, `#review`, `#decisions` | `dashboard/static/` |

The homepage's wafer images are real validation wafers, rendered by
`python scripts/export_site_images.py`; `site/img/app-operations.png` is a screenshot of `/app/?t=330`.
The per-class chart on the homepage reads `/api/model`, so it always matches the deployed model.

## Views

The app has a left sidebar (icon rail on narrow screens, bottom bar on phones) and these pages:

| Page | What it shows | Source |
|---|---|---|
| Overview | KPIs, per-tool defect bars, inspection feed with filters (search, tool, pattern, defects / review), active alerts, latest decisions | `/api/state`, `/api/wafers` |
| Alerts | Each excursion with likely cause and source, exposure, the evidence wafers, and hold / ticket / dismiss / ask copilot | `wafer.factory` (alert rule, root causes, impact) |
| Tools | Per tool: wafers, defect rate, review count, defective wafers per lot (one 0–25 scale for all tools), defect pattern mix | `/api/state`, `/api/wafers` |
| Lots | Every lot with defect rate, top pattern and status (normal, in alert, at risk after an alert, held); click a lot for its wafers | `/api/wafers`, alerts and decisions |
| Review queue | Wafers the model sent to a person; confirm or reject (kept in the browser tab only) | `needs_review` on the stream |
| Decisions | Audit log of alert actions with who made them (the demo sign-in name) and the line time | `/api/decisions` (in memory) |
| Wafer inspector | Map; line prediction from the scored stream; optional live CNN / YOLO opinions (not the alert source). Attention heatmap is CNN-only. | `wafer.infer`, `wafer.explain`, `wafer.classifiers` |
| Copilot (chat bubble on every page) | Questions answered from line data up to the submitted `as_of` (shown on each answer) | `wafer.agent.WaferAgent` (AMD Token Factory) |

Model metrics and charts are no longer a page in the app: see `docs/results/MODEL_CARD.md` and
`docs/results/report/`. `/api/model` still serves the homepage's per-class chart.

Experimental YOLO-cls (`models/yolo-cls-exp`): uncalibrated; no validated automatic-acceptance /
review policy; private training split overlaps the val demo — not a fair CNN comparison.
Separate stream: `STREAM_CSV=streams/stream_val_yolo.csv` (agent history follows the stream's
`classifier` column).

## Configuration

| Variable | Default |
|---|---|
| `WAFER_MODEL_DIR` | `models/v0-ens3` (recommended model; also used by the agent's CNN backend) |
| `STREAM_CSV` | `streams/stream_val_ens3.csv` (val stream scored by v0-ens3) |

Rebuild the stream and map pack after changing the model or stream:

```bash
WAFER_MODEL_DIR=models/v0-ens3 python scripts/make_stream.py --out streams/stream_val_ens3.csv
python scripts/export_stream_maps.py --stream streams/stream_val_ens3.csv
```

Decisions are kept in memory for the demo and reset on restart. API docs: `/api/docs`.
