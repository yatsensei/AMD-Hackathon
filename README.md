# Wafer-map yield copilot

AMD Developer Hackathon ACT III — Track 1 "Intelligent Industry".

- **Spec (source of truth for Phases 1–2):** [`docs/specs/SPEC_phase1_phase2.md`](docs/specs/SPEC_phase1_phase2.md)
- **Why the spec looks the way it does:** [`docs/specs/REVIEW_phase1_phase2.md`](docs/specs/REVIEW_phase1_phase2.md)
- **What's done:** [`docs/STATUS.md`](docs/STATUS.md)

## Phase 1 — build the dataset

```bash
pip install -r requirements.txt
export KAGGLE_USERNAME=... KAGGLE_KEY=...        # never commit these
pip install kaggle && scripts/download_data.sh  # checks the SHA-256 recorded in docs/results/data_report.md

python scripts/prepare_data.py                    # critical path: data/processed/wafers_s64.npz + metadata.parquet
python scripts/prepare_data.py --sizes 64 96 --yolo   # plus the non-critical outputs
pytest                                            # synthetic tests always; real-data tests once data/ exists
```

The first person to run `prepare_data.py` on the real file commits `docs/results/data_report.md`. It records the file's SHA-256, and everyone else's run refuses to unpickle a file that doesn't match it.

If the split search can't meet the minimum per-class counts (rare classes concentrated in few lots), it says how far off the closest seed was. Lower the minimum with `--min-count Near-full=N` and add a line to the spec's Decisions Log.
