# YOLO wafer defect classifier: getting started

## Approach
WM-811K gives one label per wafer and no boxes, and real wafers have heavy background
noise, so auto-generated boxes were unreliable. We use **YOLO classification**
(`yolo11n-cls`): it learns from the wafer label alone. "Where on the wafer" is added at
prediction time by `infer.py`, which finds the area with the densest failures.

## Step 1: Build the dataset (laptop, no GPU needed)
```bash
pip install "pandas<3" opencv-python-headless numpy
python prepare_yolo_data.py --pkl LSWMD.pkl --out yolo_cls
```
This creates `yolo_cls/train|val|test/<class>/*.png` (9 classes including "none"),
plus `yolo_cls/index.csv` with lot names for Person 3's factory simulation.

## Step 2: Install Ultralytics
On a Mac or any normal machine:
```bash
pip install ultralytics
```
On an AMD ROCm notebook, don't let pip replace the ROCm PyTorch:
```bash
pip install ultralytics --no-deps
pip install pyyaml matplotlib scipy polars psutil py-cpuinfo ultralytics-thop opencv-python-headless
python -c "import torch; print(torch.version.hip)"   # must print a version, not None
```

## Step 3: Smoke test (a couple of minutes)
```bash
yolo classify train model=yolo11n-cls.pt data=yolo_cls imgsz=128 epochs=1 batch=64 fraction=0.1 device=mps
```
Use `device=mps` on a Mac with Apple Silicon, `device=0` on an AMD GPU and `device=cpu` otherwise.
If it crashes during "AMP checks", add `amp=False`.

## Step 4: Real training
```bash
yolo classify train model=yolo11n-cls.pt data=yolo_cls imgsz=128 epochs=30 batch=128 \
     fliplr=0.5 flipud=0.5 degrees=180 hsv_h=0 hsv_s=0 hsv_v=0 erasing=0 patience=10 device=mps
```
- Flips and rotations are on because a defect pattern means the same thing in any orientation.
- Colour changes and random erasing are off, because colour carries meaning (pass vs fail) and erasing could wipe out the defect.
- If you're short on time, use 15 epochs. Results go to `runs/classify/train/`.

## Step 5: Check results
```bash
yolo classify val model=runs/classify/train/weights/best.pt data=yolo_cls split=test
```
Keep `confusion_matrix_normalized.png` and the top-1 accuracy for the pitch.
Expect Near-full (149 wafers) and Donut (555) to be the weakest classes, and say so.

## Step 6: Hand it to the agent
```python
from infer import WaferClassifier
clf = WaferClassifier("runs/classify/train/weights/best.pt")
clf.predict(wafer_map)
# {"pattern": "Edge-Loc", "confidence": 0.93, "location": "upper-right edge",
#  "fail_rate": 0.31, "top3": [...]}
```
Person 2 wraps `clf.predict` as the agent's `classify_wafer` tool.