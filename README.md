# Wafer-map yield copilot

AMD Developer Hackathon ACT III — Track 1 "Intelligent Industry".

- **Spec (Phases 1–2):** [`docs/specs/SPEC_phase1_phase2.md`](docs/specs/SPEC_phase1_phase2.md)
- **What's done:** [`docs/STATUS.md`](docs/STATUS.md) · **CNN results:** [`docs/results/OVERNIGHT.md`](docs/results/OVERNIGHT.md)
- **Teammate hand-off (data, CNN, factory, agent):** [`docs/HANDOFF.md`](docs/HANDOFF.md)
- **Hackathon brief:** [`docs/HACKATHON_BRIEF.md`](docs/HACKATHON_BRIEF.md)
- **AMD / Token Factory notes:** [`docs/AMD_SETUP.md`](docs/AMD_SETUP.md)

**Working today:** Phases 1–2 (data + CNN), Phase 4 (factory stream/alerts), **Phase 5 (Token Factory agent + CNN tools)**.  
**Pending:** YOLO adapter (Phase 3), Streamlit dashboard (Phase 6), final test-set eval, end-to-end demo polish (Phase 7).

## Phase 5 — run the agent

```bash
pip install -r requirements.txt torch torchvision   # torch needed to score wafers with the CNN
cp .env.example .env                                 # set AMD_API_KEY, AMD_BASE_URL, AMD_MODEL
# AMD_MODEL must be an id from GET $AMD_BASE_URL/models — do not guess

# Once: build maps if needed, then score the committed unscored val stream
scripts/download_data.sh && python scripts/prepare_data.py   # → data/interim/labeled.pkl
python -m wafer.agent --score-stream streams/stream_val_unscored.csv -o streams/stream_val.csv

# Ask (stdout = answer only; use --debug for timings)
python -m wafer.agent -q "What is happening on ETCH-04?" \
  --stream streams/stream_val.csv --as-of "2026-10-13 04:10"
```

```python
from wafer.agent import WaferAgent
import pandas as pd
agent = WaferAgent.from_env(pd.read_csv("streams/stream_val.csv"), as_of="2026-10-13 04:10")
print(agent.ask("What exposure has been observed?")["answer"])
```

Classifier backends: `from wafer.classifiers import get_classifier` — **cnn** default; **yolo** raises “not available yet” (no silent CNN fallback). Details: [`docs/HANDOFF.md`](docs/HANDOFF.md) §3b.

## On an AMD GPU (optional — CNN was trained on CPU)

**Guide:** [`docs/AMD_SETUP.md`](docs/AMD_SETUP.md). The live AMD component of the demo is the **agent LLM on Token Factory**, not CNN training.

```bash
git clone <this repo> && cd AMD-Hackathon
export KAGGLE_API_TOKEN=...                      # never commit it
scripts/amd/start_container.sh
# inside: scripts/amd/bootstrap.sh && tmux new -s e0 'scripts/amd/run_e0.sh'
```

Other CNN entry points: `python -m wafer.train --config <yaml>`, `python -m wafer.export`, `python -m wafer.evaluate --model models/vN` (val) / `--final` (test, once).

## Phase 1 — build the dataset

```bash
pip install -r requirements.txt
export KAGGLE_API_TOKEN=...                      # never commit it
pip install "kaggle>=2" && scripts/download_data.sh

python scripts/prepare_data.py                    # wafers_s64.npz + metadata.parquet + labeled.pkl
pytest                                            # synthetic always; real-data tests once data/ exists
```

The first person to run `prepare_data.py` on the real file commits `docs/results/data_report.md` (SHA-256 of `LSWMD.pkl`).
