"""
Train YOLO11s Single-Object Known Detector (Stream B)
Target Class: 'bicycle'
Pretrained Base: yolo11s.pt
Hardware: GPU (RTX 3050, device=0)
"""

import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO


def train_stream_b_model():
    workspace_root = Path(__file__).resolve().parent.parent
    data_yaml = workspace_root / "datasets" / "single_object_dataset" / "data.yaml"
    weights_dir = workspace_root / "stream_b" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stream B: Training Initial Known YOLO Model (yolo11s)")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Device: GPU 0 (NVIDIA RTX 3050)")
    print("=" * 60)

    # Base model requested: YOLO 11 s
    base_model = "yolo11s.pt"
    # Check if local yolo11s.pt exists in FYP_phase1
    alt_yolo11s = Path("C:/FYP_phase1/Sefl-Evolving-Video-Analytics-System/yolo11s.pt")
    if alt_yolo11s.exists():
        base_model = str(alt_yolo11s)

    model = YOLO(base_model)

    # Train for 30 epochs with batch size 16 on GPU with patience 20
    results = model.train(
        data=str(data_yaml),
        epochs=30,
        patience=20,
        imgsz=640,
        batch=16,
        device=0,
        project=str(workspace_root / "runs" / "stream_b"),
        name="known_bicycle_yolo11s",
        exist_ok=True,
        verbose=True,
        save=True
    )

    # Best weights path
    best_weights_path = workspace_root / "runs" / "stream_b" / "known_bicycle_yolo11s" / "weights" / "best.pt"
    target_weights_path = weights_dir / "known_single_object.pt"

    if best_weights_path.exists():
        shutil.copy2(best_weights_path, target_weights_path)
        print(f"\n[OK] Trained weights successfully saved to: {target_weights_path}")
    else:
        print(f"\n[WARN] best.pt not found at {best_weights_path}, checking last.pt...")
        last_weights = workspace_root / "runs" / "stream_b" / "known_bicycle_yolo11s" / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy2(last_weights, target_weights_path)
            print(f"[OK] Saved {last_weights} to {target_weights_path}")

    # Evaluate on the validation set ("evaluation dataset should tell the truth")
    print("\n" + "=" * 60)
    print("EVALUATING ON VALIDATION DATASET")
    print("=" * 60)

    val_model = YOLO(str(target_weights_path))
    val_metrics = val_model.val(data=str(data_yaml), device=0, split="val")

    print("\n--- Model Validation Metrics ---")
    print(f"Precision: {val_metrics.box.mp:.4f}")
    print(f"Recall:    {val_metrics.box.mr:.4f}")
    print(f"mAP@50:    {val_metrics.box.map50:.4f}")
    print(f"mAP@50-95: {val_metrics.box.map:.4f}")

    # Run inference test on validation images and negative images
    eval_out_dir = workspace_root / "outputs" / "stream_b_eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    val_images = list((workspace_root / "datasets" / "single_object_dataset" / "images" / "val").glob("*.jpg"))
    sample_images = val_images[:10]

    print(f"\nSaving visual evaluation predictions to {eval_out_dir}...")
    for img_p in sample_images:
        res = val_model.predict(source=str(img_p), conf=0.40, device=0, save=True, project=str(eval_out_dir), name="predictions", exist_ok=True)

    print("\n[OK] Training, validation, and evaluation completed successfully!")
    return target_weights_path


if __name__ == '__main__':
    train_stream_b_model()
