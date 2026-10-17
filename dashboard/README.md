# Yield Copilot dashboard

FastAPI backend plus a static single-page UI (no build step). It replays the scored
validation stream as a live etch line, so every screen shows real WM-811K wafers.

```bash
pip install -r requirements.txt -r dashboard/requirements.txt
uvicorn dashboard.server:app --port 8000        # from the repo root
# open http://localhost:8000   (add ?t=400 to start after the ETCH-04 excursion)
```

Works without the 2 GB dataset: the 500 stream maps ship in `dashboard/data/stream_maps.npz`
(also used by `classify_wafer` via `wafer.data.access.get_wafer_map`).
The copilot tab needs `AMD_API_KEY`, `AMD_BASE_URL` and `AMD_MODEL` (see `.env.example`);
everything else runs offline.

**Hugging Face Docker Space:** see [`docs/DEPLOY_SPACE.md`](../docs/DEPLOY_SPACE.md)
(`uvicorn` on `0.0.0.0:7860`).

## Views

| View | What it shows | Source |
|---|---|---|
| Operations | KPIs, per-tool defect bars, inspection feed, alerts with likely cause, exposure and engineer actions (hold, ticket, dismiss) | `wafer.factory` (alert rule, root causes, impact) |
| Wafer inspector | Map, model attention with the defect box, calibrated probabilities, CNN vs YOLO opinion | `wafer.infer`, `wafer.explain`, `wafer.classifiers` |
| Copilot | Questions answered from line data up to the current line time | `wafer.agent.WaferAgent` (AMD Token Factory) |
| Model quality | Validation metrics, per-class table, every report chart | `docs/results/<model>_val/metrics.json`, `docs/results/report/` |

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
