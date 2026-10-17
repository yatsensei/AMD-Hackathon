"""
Train the YOLO wafer-pattern classifier, then score it on the held-out test set.

    python train.py --smoke        # quick 1-epoch check on 10% of the data
    python train.py                # full training (30 epochs)
    python train.py --model yolo26s-cls.pt --name small   # try the bigger "small" model

Picks the device automatically: AMD GPU (ROCm) or NVIDIA -> "0", Apple Silicon -> "mps", else CPU.
Results go to runs/classify/<name>/ ; the model to use is runs/classify/<name>/weights/best.pt
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

FALLBACK_MODEL = "yolo11n-cls.pt"  # used if the YOLO26 weights can't be downloaded
RUNS = Path("runs/classify").resolve()


def pick_device():
    if torch.cuda.is_available():  # true on AMD ROCm GPUs too
        kind = "AMD ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        print(f"Using GPU ({kind}): {torch.cuda.get_device_name(0)}")
        return "0"
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (mps)")
        return "mps"
    print("No GPU found, using CPU (slow)")
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="yolo_cls", help="folder made by prepare_yolo_data.py")
    ap.add_argument("--model", default="yolo26n-cls.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--name", default="wafer")
    ap.add_argument("--device", default=None, help="override: 0, mps or cpu")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 10%% of data, just to test")
    args = ap.parse_args()

    device = args.device or pick_device()
    epochs, fraction, name = (1, 0.1, args.name + "_smoke") if args.smoke else (args.epochs, 1.0, args.name)

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Could not load {args.model} ({e}); falling back to {FALLBACK_MODEL}")
        model = YOLO(FALLBACK_MODEL)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=epochs,
        batch=args.batch,
        fraction=fraction,
        device=device,
        project=str(RUNS),  # absolute path, so Ultralytics doesn't nest it inside its own runs folder
        name=name,
        exist_ok=True,
        patience=10,         # stop early if validation stops improving
        # A defect pattern means the same thing in any orientation -> flips and rotations are safe.
        fliplr=0.5,
        flipud=0.5,
        degrees=180,
        # Colour carries meaning (pass vs fail) and erasing could remove the defect -> turn these off.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,
        plots=True,
    )

    best = Path(model.trainer.best)  # ask Ultralytics where it actually saved the model
    print(f"\nScoring {best} on the held-out TEST set...")
    metrics = YOLO(str(best)).val(data=args.data, split="test", imgsz=args.imgsz, device=device,
                                  project=str(RUNS), name=name + "_test", exist_ok=True)
    print(f"\nTest top-1 accuracy: {metrics.top1:.3f}   top-5: {metrics.top5:.3f}")
    print(f"Model:            {best}")
    print(f"Confusion matrix: {Path(metrics.save_dir) / 'confusion_matrix_normalized.png'}")


if __name__ == "__main__":
    main()