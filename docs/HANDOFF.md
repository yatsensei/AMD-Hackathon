# Hand-off: using the data and the CNN from other phases

Phases 1–2 (data + CNN) are done. This page is for the YOLO, factory-simulation, agent and dashboard work: what you can use today, and the rules that keep our numbers honest.

## 1. Setup (once, about 10 minutes, any laptop, no GPU)
```bash
git clone -b claude/great-ptolemy-wdot0b https://github.com/yatsensei/AMD-Hackathon && cd AMD-Hackathon
pip install -r requirements.txt torch torchvision
scripts/download_data.sh            # public Kaggle archive, no login needed (or set KAGGLE_API_TOKEN)
python scripts/prepare_data.py      # ~7 min: builds data/processed/* and data/interim/labeled.pkl
```
If you only need the split (lot, wafer index, label, split), skip the download: `splits/wafer_split_v1.csv.gz` is in git.

## 2. Classifier API (agent: Phase 5 · dashboard: Phase 6 · factory: Phase 4)
```python
from wafer.infer import classify, classify_batch, explain     # loads models/v1, else models/v0
r = classify(wafer_map)            # wafer_map: 2D array, 0 = no die, 1 = pass, 2 = fail, any size
# {"pattern": "Edge-Ring", "confidence": 0.94, "probs": {...9 classes...},
#  "top2": [["Edge-Ring", 0.94], ["Donut", 0.03]], "defect_prob": 0.97,
#  "needs_review": False, "review_reasons": [], "model_version": "v0"}
e = explain(wafer_map)             # {"heatmap": HxW float32 in [0,1], "bbox": [x0,y0,x1,y1] | None, "pattern": ...}
```
- **About 4–10 ms per wafer on a laptop CPU** (`models/v0` single CNN: 4 ms; `models/v0-ens3`: 9.5 ms). Pick a model with `WAFER_MODEL_DIR=models/v0-ens3`.
- Plain Python types, so `json.dumps(r)` works for agent tool outputs.
- Invalid input (wrong values, empty map) raises `ValueError` with a clear message.
- `wafer/infer_mock.py` has the same functions and keys if you need a stand-in.

**Agent (Phase 5):**
- When `needs_review` is True, **say the model is unsure** and use `review_reasons`:
  - `low_confidence`: below the calibrated threshold;
  - `confusable_pair`: e.g. Edge-Loc vs Scratch with a small margin;
  - `possible_defect`: "none" was most likely, but the defect-recall safety rule flagged it.
- Never turn a 0.4 confidence into "likely".
- `confidence` is calibrated: 0.9 means right about 90% of the time on validation.

**Dashboard (Phase 6):**
- `from wafer.viz import render_wafer, render_heatmap`, so wafers look the same everywhere.
- `render_heatmap(map, e["heatmap"], e["bbox"])` gives the overlay with the box.
- Examples: `docs/results/v0_val/gradcam_examples.png`.

The box marks the fail-die cluster that drove the decision. Call it an *explanation/attribution*, not *detection*: there were no box labels to train on.

## 3. Wafers for the factory stream (Phase 4)
```python
from wafer.data.access import wafers_for_stream, get_wafer_map
df = wafers_for_stream("val")      # uid, lotName, waferIndex, label, ... ordered by lot and wafer
m = get_wafer_map(df.uid[0])       # the raw map to send through classify()
```
- **Stream `val` wafers while building the demo. Switch to `"test"` only after the single final test evaluation has run.** Otherwise we'd be looking at test predictions before the final score. Training wafers are refused: the model has seen them.
- Group them into fictional lots and tools however the story needs. Keep `uid` so every streamed wafer traces back to the dataset.
- `label` is the true pattern. Use it to inject a "drifting tool" (e.g. pick Edge-Ring wafers for Tool B), but the alert logic must react to the **model's** `classify()` output, not the label.

## 4. YOLO benchmark (Phase 3)
- PNG folders: `python scripts/prepare_data.py --yolo` writes `yolo_data/{train,val,test}/<class>/<uid>.png`. Defaults to the area-fraction rendering; add `--yolo-encoding nearest` for the plain 0/1/2 colours.
- To be scored on exactly the same split and metrics as the CNN, write a CSV `uid,prob_Center,prob_Donut,...,prob_none` for **all val and test** wafers, then:
  ```bash
  python -m wafer.evaluate --predictions path/to/yolo_preds.csv          # validation score
  ```
- The test score comes only from the single final run, where YOLO is a pre-registered comparator.

## 5. Numbers you may quote (and how)
- **Validation results:** `docs/results/OVERNIGHT.md` and `docs/results/v0_val/`. They're **validation** numbers, slightly optimistic.
- **Pitch numbers** must come from `docs/results/final/` once the final test run happens (not yet).
- **Our CNN was trained on CPU** (no GPU credits), so don't say it was trained on AMD. The AMD part of the product is the agent's LLM on Token Factory.
