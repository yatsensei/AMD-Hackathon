"""
Run the trained YOLO classifier on one wafer map and return a result the agent can use.

    from infer import WaferClassifier
    clf = WaferClassifier("runs/classify/train/weights/best.pt")
    print(clf.predict(wafer_map_array))
    # {"pattern": "Edge-Loc", "confidence": 0.93, "location": "upper-right edge",
    #  "fail_rate": 0.31, "top3": [...]}

The location is NOT from the model. It comes from a simple density check: smooth the
map of failed chips and find the area where failures are most concentrated. It's only
used to make the explanation more specific.
"""
import cv2
import numpy as np

from prepare_yolo_data import render

WHOLE_WAFER = {"Near-full", "Random"}


def where_on_wafer(cx, cy):
    """Plain-English location from a 0-1 position, e.g. 'upper-right edge' or 'centre'."""
    dx, dy = cx - 0.5, cy - 0.5
    r = (dx ** 2 + dy ** 2) ** 0.5
    if r < 0.15:
        return "centre"
    vert = "upper" if dy < -0.12 else "lower" if dy > 0.12 else ""
    horiz = "left" if dx < -0.12 else "right" if dx > 0.12 else ""
    side = "-".join(p for p in (vert, horiz) if p) or "middle"
    return f"{side} {'edge' if r > 0.35 else 'area'}"


def densest_failure_spot(small):
    """Centre (0-1 coords) of the area where failed chips are most concentrated."""
    size = small.shape[0]
    valid = (small > 0).astype(np.float32)
    fail = (small == 2).astype(np.float32)
    k = max(3, size // 8) | 1  # odd kernel, ~1/8 of the wafer
    # Fail fraction among real chips nearby; dividing by `valid` keeps the wafer edge fair.
    density = cv2.blur(fail, (k, k)) / np.maximum(cv2.blur(valid, (k, k)), 1e-6)
    density[valid == 0] = 0
    hot = (density >= max(0.5, density[valid > 0].mean() + 0.25)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(hot, connectivity=8)
    if n <= 1:  # no clear hotspot: fall back to the single densest point
        y, x = np.unravel_index(density.argmax(), density.shape)
        return x / size, y / size
    biggest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    cx, cy = cents[biggest]
    return cx / size, cy / size


class WaferClassifier:
    def __init__(self, weights, size=128):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.size = size

    def predict(self, wafer_map):
        small, img = render(np.asarray(wafer_map), self.size)
        res = self.model.predict(img, imgsz=self.size, verbose=False)[0]
        names, probs = res.names, res.probs
        pattern = names[int(probs.top1)]
        top3 = [{"pattern": names[int(i)], "confidence": round(float(c), 3)}
                for i, c in zip(probs.top5[:3], probs.top5conf[:3].tolist())]

        valid = small > 0
        fail_rate = float((small[valid] == 2).mean()) if valid.any() else 0.0
        if pattern == "none":
            location = None
        elif pattern in WHOLE_WAFER:
            location = "whole wafer"
        elif pattern == "Edge-Ring":
            location = "all around the edge"
        else:
            location = where_on_wafer(*densest_failure_spot(small))

        return {
            "pattern": pattern,
            "confidence": round(float(probs.top1conf), 3),
            "location": location,
            "fail_rate": round(fail_rate, 3),
            "top3": top3,
        }