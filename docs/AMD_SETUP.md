# Running the CNN on AMD — step by step

Everything that trains or evaluates the CNN runs on the AMD GPU. You do the AMD side. The code is already written and tested, so these steps are mostly "run this, check that, send back the files".

There are two ways in, depending on what the AMD access gives you:
- **A. A Jupyter notebook with a GPU** (notebooks.amd.com / token factory). Most likely, per the brief.
- **B. A VM/droplet with SSH and Docker** (AMD Developer Cloud GPU instance).

---

## What Token Factory gives us (from its docs, 14 Oct)
- **A shared team API key** (`rc-...`) for free LLM endpoints (DeepSeek, Qwen, GLM, Gemma…) at `https://developer.amd.com.cn/radeon/api/v1`. It's for the **agent** teammates, not the CNN. Hand it over privately and keep it in an env var (`RADEON_API_KEY`, see `.env.example`). Never commit it or paste it into chat, a notebook output or a screenshot. All 5 of us share one key: 30 requests/min and 8 concurrent, team-wide.
- **GPU instances (shell or notebook)** for anything the catalog doesn't offer, like training our CNN. These are paid with **credits**, so check your balance and **stop the instance when you're done** (`DELETE /api/notebook/current` or the stop button). This is where the steps below run.
- The service is **AMD Radeon Cloud**, so the GPU may be a Radeon card rather than an Instinct MI300X. The code runs on either (ROCm + PyTorch). The pitch must name **whatever `docs/results/env_amd.json` records**, not "MI300X" by default.

## 0. Before you start (once, ~10 min)
1. **Kaggle API token.** Go to kaggle.com → *Settings* → *API* → *Create New Token*. You get `kaggle.json` with a username and key. Keep it private: never paste it into chat, a notebook output, or git.
2. **Accept the dataset.** Open https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map while logged in, so the API is allowed to download it.
3. **Repo access from the AMD machine.** The notebook clones `https://github.com/yatsensei/AMD-Hackathon`. The submission needs a public repo anyway, so making it public now is simplest. If it stays private, you need a GitHub token with read access (and write access if you want to push results from the AMD side).
4. **AMD access.** At notebooks.amd.com/tokenfactory every team member must sign in and click *Redeem API Key* (the key appears only after all 5 have). Then find the **instance / notebook** option (templates list) and check your credits. Post the answers in team chat (brief §4.2):
   - Is there a GPU? Which one?
   - Is there a terminal (JupyterLab *File → New → Terminal*)?
   - Does disk persist between sessions?
   - Can it reach the internet (github.com, kaggle.com, pypi.org)?

## A. Jupyter notebook path (most likely)
1. In Token Factory, launch a **notebook instance** with a **PyTorch / ROCm** template (pick the one with a GPU). Open its JupyterLab.
2. Upload `notebooks/amd_e0.ipynb` from this repo, or open a terminal and `git clone` the repo, then open the notebook from there.
3. **Run cell 1 (GPU check).** You must see `hip: <version>` (not `None`) and an AMD device name (Instinct or Radeon). Note which one. If `hip` is `None`, you're on a CPU or NVIDIA build: switch to the ROCm image/kernel. **Never** `pip install torch` from PyPI there, because that installs the CUDA build.
4. **Run cells 2–3.** They get the code, install dependencies (torch is left alone), and check the GPU, including a convolution through MIOpen. Cell 3 must end with `PASS`.
5. **Run cell 4.** Type your Kaggle username and key. They stay in memory only.
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
export KAGGLE_USERNAME=... KAGGLE_KEY=...      # in your shell only
scripts/amd/start_container.sh                # ROCm PyTorch container, repo mounted at /workspace
# inside the container:
scripts/amd/bootstrap.sh                      # deps + GPU check (must print PASS)
tmux new -s e0 'scripts/amd/run_e0.sh'        # data -> 5 training runs -> models/v0 -> val report
```
If your instance already has ROCm PyTorch installed without Docker, skip `start_container.sh` and run `bootstrap.sh` directly. It refuses to continue if torch has no ROCm support.

**When finished: stop the instance** so it stops using credits.

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
- **`kaggle: 403`:** accept the dataset's terms on the Kaggle page (step 0.2). **`401`:** wrong username or key.
- **Kaggle or GitHub unreachable from the AMD machine:** download `LSWMD.pkl` elsewhere and upload it to `data/raw/` (2 GB, so use the terminal with `scp` or `curl` from your own storage if the browser upload struggles). Then skip the download step.
- **Split search error "No split seed … meets the minimum per-class counts":** it prints how far off it was. Lower the minimum, e.g. `python scripts/prepare_data.py --min-count Near-full=10`, and tell Claude the numbers.
- **Out of GPU memory with 5 parallel runs:** `N=2 bash scripts/run_sweep.sh configs/e0_baseline.yaml`.
- **Anything else:** send the last 30 lines of the failing log (`experiments/logs/*.log`, or the cell output).
