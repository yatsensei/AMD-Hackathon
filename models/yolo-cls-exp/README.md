# Experimental model: `yolo-cls-exp`

**Status: experimental second opinion — not a fair comparator to the CNN.**

| Field | Value |
|---|---|
| Task | Ultralytics YOLO **classification** (`yolo26n-cls`) |
| Weights | `best.pt` (~3.2 MB) |
| Input | 128×128 nearest-neighbour resize, BGR colour render (pass/fail) |
| Classes | Same 9 names as `wafer.constants.CLASSES` (verified against checkpoint `names`) |
| Calibration | **None** — confidence is raw softmax |
| Review rules | **Not validated** — adapter never sets `needs_review` from CNN τ/θ |

## Training / split honesty

This checkpoint was trained on a **private** 70/15/15 stratified-by-label sample
(`prepare_yolo_data.py --mode cls`, seed 0, class caps), **not** on
`splits/wafer_split_v1.csv.gz`. Against the shared split, roughly half of the
training rows land in our val/test lots (and vice versa). The val demo factory
stream therefore **overlaps** wafers/lots the YOLO model may have seen in training.

Do **not** claim:

- fair comparison with `models/v0-ens3`
- held-out / lot-grouped performance
- independent confirmation of CNN alerts

Use only as an experimental selectable backend (`WAFER_CLASSIFIER=yolo`) and a
separately scored stream (`streams/stream_val_yolo.csv`).

## Serve

```bash
pip install ultralytics
from wafer.classifiers import get_classifier
clf = get_classifier("yolo")
clf.classify(wafer_map)   # classifier="yolo", model_version="yolo-cls-exp"
```
