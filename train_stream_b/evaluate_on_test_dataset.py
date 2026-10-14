"""
Evaluate the trained Stream B YOLO11s model on the downloaded test/validation dataset.
Generates comprehensive quantitative metrics and per-image detection breakdown.
"""

from pathlib import Path
import json
from ultralytics import YOLO

workspace_root = Path(__file__).resolve().parent.parent


def evaluate_dataset():
    weights_path = workspace_root / "stream_b" / "weights" / "known_single_object.pt"
    data_yaml = workspace_root / "datasets" / "single_object_dataset" / "data.yaml"

    print("=" * 65)
    print("STREAM B: QUANTITATIVE EVALUATION ON DOWNLOADED DATASET")
    print(f"Weights: {weights_path}")
    print(f"Dataset YAML: {data_yaml}")
    print("=" * 65)

    model = YOLO(str(weights_path))

    # Run formal YOLO validation on the validation split
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        device=0,
        conf=0.40,
        iou=0.45,
        verbose=False
    )

    print("\n--- VALIDATION METRICS SUMMARY ---")
    print(f"Precision (P):       {metrics.box.mp * 100:.2f}%")
    print(f"Recall (R):          {metrics.box.mr * 100:.2f}%")
    print(f"mAP@50:              {metrics.box.map50 * 100:.2f}%")
    print(f"mAP@50-95:           {metrics.box.map * 100:.2f}%")
    print(f"Inference Latency:   {metrics.speed.get('inference', 0.0):.2f} ms")

    # Inspect per-image predictions on test/val set
    val_images = sorted(list((workspace_root / "datasets" / "single_object_dataset" / "images" / "val").glob("*.jpg")))
    out_eval_dir = workspace_root / "outputs" / "stream_b_eval"
    out_eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- EVALUATION ON {len(val_images)} VALIDATION IMAGES ---")
    pos_detected = 0
    neg_clean = 0
    total_pos = 50
    total_neg = 20

    for img_p in val_images:
        res = model.predict(source=str(img_p), conf=0.40, device=0, verbose=False)[0]
        num_boxes = len(res.boxes)
        # Check ground truth label
        lbl_p = workspace_root / "datasets" / "single_object_dataset" / "labels" / "val" / f"{img_p.stem}.txt"
        has_gt = lbl_p.exists() and len(open(lbl_p).readlines()) > 0

        if has_gt:
            if num_boxes > 0:
                pos_detected += 1
        else:
            if num_boxes == 0:
                neg_clean += 1

    print(f"Bicycle Images Detected:          {pos_detected}/{total_pos} ({pos_detected / total_pos * 100:.1f}%)")
    print(f"Negative Background Suppression:  {neg_clean}/{total_neg} ({neg_clean / total_neg * 100:.1f}% false-alarm free)")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_dataset()
