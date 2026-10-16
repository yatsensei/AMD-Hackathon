# Wafer-map yield copilot

AMD Developer Hackathon ACT III — Track 1 "Intelligent Industry".

- **Spec (source of truth for Phases 1–2):** [`docs/specs/SPEC_phase1_phase2.md`](docs/specs/SPEC_phase1_phase2.md)
- **Why the spec looks the way it does:** [`docs/specs/REVIEW_phase1_phase2.md`](docs/specs/REVIEW_phase1_phase2.md)
- **What's done:** [`docs/STATUS.md`](docs/STATUS.md) · **latest results:** [`docs/results/OVERNIGHT.md`](docs/results/OVERNIGHT.md)
- **For teammates (classifier API, data, YOLO scoring):** [`docs/HANDOFF.md`](docs/HANDOFF.md)
- **Hackathon brief (judging, rules):** [`docs/HACKATHON_BRIEF.md`](docs/HACKATHON_BRIEF.md)
- **Our scope:** Phases 1–2 (data + CNN). Agent, dashboard and YOLO are owned by teammates; `wafer/infer_mock.py` shows them the API they'll get.

## On an AMD GPU (optional: no credits right now, so models are trained on CPU)

**Step-by-step guide: [`docs/AMD_SETUP.md`](docs/AMD_SETUP.md).** Notebook version: [`notebooks/amd_e0.ipynb`](notebooks/amd_e0.ipynb). Terminal version:

```bash
git clone <this repo> && cd AMD-Hackathon
export KAGGLE_API_TOKEN=...                      # on the instance, never committed
scripts/amd/start_container.sh                  # ROCm PyTorch container, repo at /workspace (re-run to re-attach)
# inside the container:
scripts/amd/bootstrap.sh                        # deps (never touches torch), GPU check -> docs/results/env_amd.json
tmux new -s e0 'scripts/amd/run_e0.sh'          # data -> E0 x5 seeds in parallel -> models/v0 -> val report
```

Then commit `docs/results/` (leaderboard, `v0_val/`, `data_report.md`, `env_amd.json`) and `models/v0/`, so everyone (including Claude) can read the results without GPU access.

Other entry points: `python -m wafer.train --config <yaml> [--seed N]`, `scripts/run_sweep.sh configs/tier1/*.yaml`, `python -m wafer.export --runs <runs> --version vN [--tta]`, `python -m wafer.evaluate --model models/vN` (val) / `--final` (test, once), `scripts/bench_latency.py --model models/vN`.

## For the agent and dashboard (Phases 5–6)

```python
from wafer.classifiers import get_classifier   # cnn default; yolo not available yet
from wafer.infer import classify, classify_batch, explain
from wafer.viz import render_wafer, render_heatmap
from wafer.agent import WaferAgent              # Token Factory LLM + tools
```

Agent setup: set `AMD_API_KEY`, `AMD_BASE_URL`, `AMD_MODEL` in `.env` (see `.env.example`).  
Run: `python -m wafer.agent --score-stream streams/stream_val_unscored.csv -o streams/stream_val.csv` then  
`python -m wafer.agent -q "..." --stream streams/stream_val.csv --as-of "2026-10-12 18:00"`.  
Details: [`docs/HANDOFF.md`](docs/HANDOFF.md) §3b.

## Phase 1 — build the dataset

```bash
pip install -r requirements.txt
export KAGGLE_API_TOKEN=...                      # never commit it
pip install "kaggle>=2" && scripts/download_data.sh  # checks the SHA-256 recorded in docs/results/data_report.md

python scripts/prepare_data.py                    # critical path: data/processed/wafers_s64.npz + metadata.parquet
python scripts/prepare_data.py --sizes 64 96 --yolo   # plus the non-critical outputs
pytest                                            # synthetic tests always; real-data tests once data/ exists
```

The first person to run `prepare_data.py` on the real file commits `docs/results/data_report.md`. It records the file's SHA-256, and everyone else's run refuses to unpickle a file that doesn't match it.

If the split search can't meet the minimum per-class counts (rare classes concentrated in few lots), it says how far off the closest seed was. Lower the minimum with `--min-count Near-full=N` and add a line to the spec's Decisions Log.
