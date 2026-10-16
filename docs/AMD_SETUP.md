# Running the CNN on AMD — step by step

> **Status (15 Oct): the team has no GPU credits, so the CNN is trained on CPU** (`configs/e0_cpu.yaml` and `configs/cpu/`). See `docs/HANDOFF.md` and `docs/results/OVERNIGHT.md`. This page stays in case an AMD GPU becomes available: the same code runs there unchanged and would make "trained on AMD" true.

Everything that trains or evaluates the CNN runs on the AMD GPU. You do the AMD side. The code is already written and tested, so these steps are mostly "run this, check that, send back the files".

There are two ways in, depending on what the AMD access gives you:
- **A. A Jupyter notebook with a GPU** (notebooks.amd.com / token factory). Most likely, per the brief.
- **B. A VM/droplet with SSH and Docker** (AMD Developer Cloud GPU instance).

---

## What Token Factory gives us (checked 14 Oct; agent live Phase 5)
- **Only an LLM API:** one shared team key (`rc-...`) and a rotating catalog of free models at `https://developer.amd.com.cn/radeon/api/v1`. That is the **live AMD component** of the product (models run on AMD GPUs). The Phase 5 agent uses it via the OpenAI-compatible SDK.
- **Env vars** (see `.env.example`; never commit real values):
  - `AMD_API_KEY` — team key (optional legacy alias: `RADEON_API_KEY`)
  - `AMD_BASE_URL` — `https://developer.amd.com.cn/radeon/api/v1`
  - `AMD_MODEL` — pick an id from `GET $AMD_BASE_URL/models` (catalog changes; do not hard-code)
- Shared team limits: ~30 RPM, 8 concurrent, daily spend cap. Honour `Retry-After` on 429.
- **Run the agent:** [`docs/HANDOFF.md`](HANDOFF.md) §3b / `python -m wafer.agent --help`.
- **No GPU instances in our UI** for CNN training. Options if you want AMD training later:
  1. **AMD Developer Cloud** (Instinct MI300X / ROCm) — paths **A** / **B** below.
  2. **Teammate AMD GPU** with ROCm — path **B** without Docker.
  3. **CPU training** (what we shipped for `models/v0*`) — pitch must **not** say the CNN was trained on AMD; Token Factory covers the AMD claim for the agent.

## 0. Before you start (once, ~10 min)
1. **Kaggle API token.** kaggle.com → *Settings* → *API Tokens* → create one (the new kind; the legacy `kaggle.json` isn't needed). The scripts read it from the `KAGGLE_API_TOKEN` environment variable. Keep it private: never paste it into chat, a notebook output, or git.
2. **Accept the dataset.** Open https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map while logged in, so the API is allowed to download it.
3. **Repo access.** The repo is public, so cloning needs no token. Pushing results back needs a GitHub token with write access (or just download the results bundle).
4. **A GPU** from the options above.

## A. Jupyter notebook path (most likely)
1. On the GPU machine (e.g. AMD Developer Cloud), open JupyterLab with a **PyTorch / ROCm** image.
2. Upload `notebooks/amd_e0.ipynb` from this repo, or open a terminal and `git clone` the repo, then open the notebook from there.
3. **Run cell 1 (GPU check).** You must see `hip: <version>` (not `None`) and an AMD device name (Instinct or Radeon). Note which one. If `hip` is `None`, you're on a CPU or NVIDIA build: switch to the ROCm image/kernel. **Never** `pip install torch` from PyPI there, because that installs the CUDA build.
4. **Run cells 2–3.** They get the code, install dependencies (torch is left alone), and check the GPU, including a convolution through MIOpen. Cell 3 must end with `PASS`.
5. **Run cell 4.** Paste your Kaggle API token. It stays in memory only.
6. **Run cell 5** (about 5–10 min, CPU). Downloads `LSWMD.pkl` (about 2 GB) and builds the dataset. Read the data report it prints, and check three things:
   - the labelled count is about 172,950;
   - every class has wafers in val and test;
   - the **lots per class** for Near-full and Donut. Post those two numbers in chat.
7. **Run cell 6** (minutes). Trains the baseline 5 times in parallel on the GPU. The table at the end shows macro-F1 per run.
8. **Run cell 7.** Exports the best run to `models/v0` and evaluates it on **validation** (never test). It prints the CPU latency check.
9. **Run cell 8.** Classifies one real wafer on the GPU: this is the function the agent and dashboard call.
10. **Run cell 9.** Packs everything into `results_bundle.tar.gz`. Either download it (right-click → Download) and send it, or push it with cell 10 (needs a GitHub token with write access).

## B. SSH + Docker path
```bash
git clone -b claude/great-ptolemy-wdot0b https://github.com/yatsensei/AMD-Hackathon && cd AMD-Hackathon
export KAGGLE_API_TOKEN=...                    # in your shell only
scripts/amd/start_container.sh                # ROCm PyTorch container, repo mounted at /workspace
# inside the container:
scripts/amd/bootstrap.sh                      # deps + GPU check (must print PASS)
tmux new -s e0 'scripts/amd/run_e0.sh'        # data -> 5 training runs -> models/v0 -> val report
```
If your instance already has ROCm PyTorch installed without Docker, skip `start_container.sh` and run `bootstrap.sh` directly. It refuses to continue if torch has no ROCm support.

**When finished: stop the GPU machine** so it stops using credits.

## C. Let Claude run it (CPU, no AMD GPU needed)
In this Claude Code cloud environment's settings (environment menu in the session title bar → *Edit*):
1. **Environment variable:** `KAGGLE_API_TOKEN` = your Kaggle API token.
2. **Network access → Allowed domains:** add `www.kaggle.com`, `kaggle.com` and `api.kaggle.com` (the download itself comes from Google Cloud Storage, which is already reachable).
3. Start a new session and say "run Phase 1 and train v0". Claude downloads the data, builds the dataset, trains on CPU (a few hours), and commits the results like the AMD path would. The results say "cpu", not an AMD GPU.

## What to send back
Either `results_bundle.tar.gz`, or a push containing:
- `docs/results/env_amd.json`: proof of the AMD hardware/software (pitch AMD slide)
- `docs/results/data_report.md`: dataset facts, file SHA-256, split
- `docs/results/leaderboard.csv`: one row per training run
- `docs/results/v0_val/`: validation metrics, confusion matrix PNG, reliability and coverage plots, error gallery
- `models/v0/`: the trained CNN (`weights/m0.safetensors` and `cnn.pt`, calibration, README with latency)

Never commit `data/`, `experiments/*/weights*`, or any token.

## How the team plan maps to what runs
| Team plan step | What the code does | Where |
|---|---|---|
| 1. Download from Kaggle | `scripts/download_data.sh` (checks the file's SHA-256) | notebook cell 5 |
| 2. Load with old-pandas patch | `wafer/data/load.py` aliases old module paths automatically | cell 5 |
| 3. Flatten labels, keep labelled | `wafer/data/preprocess.py::clean` | cell 5 |
| 4. 0 = no chip, 1 = pass, 2 = fail | validated for every map; invalid/empty maps dropped and counted | cell 5 |
| 5. Resize to 64×64 nearest | **Both** encodings are saved. Nearest (`X_nearest`) as planned, plus area-fraction (`X_area`), which is the default because nearest can erase a 1-die scratch completely (measured, spec 3.5). E1 compares them | cell 5 |
| 6. 70/15/15 stratified by label | Stratified by label **and grouped by lot**, so wafers from one lot never sit in both train and test. Otherwise scores are inflated (spec Rule 0.2) | cell 5 |
| 7. Imbalance | Both: `none` downsampled to 15k per epoch (fresh each epoch) **and** √-class-weighted CE | cell 6 |
| 8. `wafers.npz` + YOLO PNGs | `data/processed/wafers_s64.npz`; PNGs with `prepare_data.py --yolo` | cell 5 |
| 9. Keep `lotName`, `waferIndex` | in `data/processed/metadata.parquet` (plus `uid`, `split`) | cell 5 |
| CNN: 3–4 conv blocks → GAP → 9, 3 channels | `small_cnn`: 4 blocks, GAP, 9 outputs, 3-channel input | cell 6 |
| Adam 1e-3, batch 256, weighted CE, 15–30 epochs | AdamW 1e-3, batch 256, weighted CE; E0 runs 60 epochs once to find where it plateaus, then later configs use that | cell 6 |
| Rotations and flips | all 8 rotations/flips (exact); small-angle rotation as an experiment | cell 6 |
| Evaluate: accuracy, per-class accuracy, confusion PNG | all of those, plus macro-F1 (headline, because 85% `none` makes accuracy misleading), CIs, calibration. **On validation now; test once at the end** | cell 7 |
| `torch.save(state_dict, "cnn.pt")` | `models/v0/cnn.pt` (plus a safetensors copy that the API loads) | cell 7 |
| `classify(map) → {"pattern", "confidence"}` | `wafer.infer.classify` returns those keys plus `probs`, `needs_review`, etc. | cell 8 |

## Troubleshooting
- **`hip: None` / no GPU:** wrong image or kernel. Pick the ROCm/PyTorch one. In Docker, the container needs `--device=/dev/kfd --device=/dev/dri` (the script adds them).
- **First epochs very slow:** MIOpen is searching for kernels. The code sets `MIOPEN_FIND_MODE=FAST`. Later runs reuse the `.miopen/` cache.
- **`kaggle: 403`:** accept the dataset's terms on the Kaggle page (step 0.2). **`401`:** wrong or expired token.
- **`unrecognized arguments: -d`:** an old copy of the script; `git pull` (kaggle 2.x takes the dataset name without `-d`).
- **Kaggle or GitHub unreachable from the AMD machine:** download `LSWMD.pkl` elsewhere and upload it to `data/raw/` (2 GB, so use the terminal with `scp` or `curl` from your own storage if the browser upload struggles). Then skip the download step.
- **Split search error "No split seed … meets the minimum per-class counts":** it prints how far off it was. Lower the minimum, e.g. `python scripts/prepare_data.py --min-count Near-full=10`, and tell Claude the numbers.
- **Out of GPU memory with 5 parallel runs:** `N=2 bash scripts/run_sweep.sh configs/e0_baseline.yaml`.
- **Anything else:** send the last 30 lines of the failing log (`experiments/logs/*.log`, or the cell output).
