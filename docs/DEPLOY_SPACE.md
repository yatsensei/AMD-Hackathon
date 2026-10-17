# Deploy as a Hugging Face Docker Space

The Sentry dashboard is packaged for a **Docker** Space. The image serves
`dashboard.server:app` with Uvicorn on `0.0.0.0:7860`, using the validation demo
stream and `models/v0-ens3`. It does **not** include the 2 GB raw WM-811K dataset;
wafer maps for the 500-stream demo ship in `dashboard/data/stream_maps.npz`.

This guide prepares and runs the container. **Do not deploy or purchase** hardware
until the team is ready — secrets stay in the host / Space settings, never in git.

## What the image contains

| Asset | Path |
|---|---|
| FastAPI + static UI | `dashboard/` |
| CNN ensemble (recommended / default) | `models/v0-ens3/` |
| Experimental YOLO-cls | `models/yolo-cls-exp/` (+ `ultralytics` in the image) |
| Scored validation streams | `streams/stream_val_ens3.csv` (CNN), `streams/stream_val_yolo.csv` (YOLO) |
| Stream wafer maps | `dashboard/data/stream_maps.npz` |
| Model-quality charts / metrics | `docs/results/report/`, `docs/results/v0-ens3_val/` |
| Factory root causes | `wafer/factory/root_causes.json` |

Excluded: `.env`, `.venv`, `/data`, other model dirs (`v0`, `v0-tta`), YOLO training
PNG dumps, test stream, training scripts, and pytest.

Default Space runtime stays on the **CNN** stream. To demo experimental YOLO:

```bash
docker run ... -e STREAM_CSV=/app/streams/stream_val_yolo.csv -e WAFER_CLASSIFIER=yolo ...
```

(The dashboard agent follows the stream's `classifier` column either way.)

## Required hosting environment variables

Set these on the Space (**Settings → Variables and secrets**) or when running Docker
locally. Prefer **secrets** for the API key.

| Variable | Required | Purpose |
|---|---|---|
| `AMD_API_KEY` | Yes (for Copilot) | Token Factory API key — never commit |
| `AMD_BASE_URL` | Yes (for Copilot) | OpenAI-compatible base URL (no trailing slash issues; trailing `/` is stripped) |
| `AMD_MODEL` | Yes (for Copilot) | Model id from `GET $AMD_BASE_URL/models` (e.g. a Qwen id) — do not guess |

Optional overrides (image defaults are already set for the demo):

| Variable | Default in image |
|---|---|
| `STREAM_CSV` | `/app/streams/stream_val_ens3.csv` (use `stream_val_yolo.csv` for YOLO demo) |
| `WAFER_MODEL_DIR` | `/app/models/v0-ens3` |
| `WAFER_CLASSIFIER` | `cnn` (set `yolo` only with the YOLO stream) |
| `WAFER_YOLO_DIR` | `/app/models/yolo-cls-exp` (optional override) |

Without `AMD_*`, Operations / wafer inspector / model quality still work; the Copilot
tab shows offline.

## Local container smoke test

```bash
# From the repo root (Docker Desktop / daemon must be running)
docker build -t yield-copilot-space .

docker run --rm -p 7860:7860 \
  -e AMD_API_KEY \
  -e AMD_BASE_URL \
  -e AMD_MODEL \
  yield-copilot-space
```

Then open http://localhost:7860 for the homepage, or http://localhost:7860/app/?t=400 to land in the dashboard after the ETCH-04 excursion.

Quick API checks:

```bash
curl -s http://127.0.0.1:7860/api/meta | python -m json.tool | head
curl -s "http://127.0.0.1:7860/api/state?as_of=2026-10-13T21:12:00" | python -c "import sys,json; s=json.load(sys.stdin); print(len(s['alerts']), s['kpis'])"
```

## Create the Space later (not done in this prep)

1. On Hugging Face: **New Space** → SDK **Docker** → pick a hardware tier when ready.
2. Push this repo (or a deploy branch) so the Space root has `Dockerfile`, `.dockerignore`,
   and a `README.md` with the YAML `sdk: docker` / `app_port: 7860` block.
3. Add `AMD_API_KEY`, `AMD_BASE_URL`, `AMD_MODEL` as Space secrets/variables.
4. Wait for the build; confirm Copilot answers on the val stream.

## Remaining blockers

| Item | Status |
|---|---|
| YOLO second opinion | Experimental YOLO-cls bundled; not a fair CNN comparator (private split) |
| Final test-set evaluation | Not run; Space must keep using **val** stream |
| HF Space creation / paid hardware | Not performed in this prep |
| AMD Token Factory network reachability from HF | Confirm after first deploy (egress to `AMD_BASE_URL`) |
| Docker Desktop / daemon | Required for local `docker build`; if stopped, start it before smoking the image |
| Experimental YOLO-cls | Bundled (`models/yolo-cls-exp` + `ultralytics`); requires matching CPU `torch`/`torchvision`. Not a fair CNN comparator (private training split). |
| Render Free 512 MB | Unlikely for Torch stack (see prior RAM inspection) |

## Repo files added for Space packaging

- `Dockerfile` — Uvicorn `dashboard.server:app` on `0.0.0.0:7860`
- `.dockerignore` — drops secrets, raw data, unused models
- `requirements-space.txt` — runtime Python deps (Torch installed in the Dockerfile)
- Root `README.md` YAML frontmatter — `sdk: docker`, `app_port: 7860`
