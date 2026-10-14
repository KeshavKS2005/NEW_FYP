"""
Prepare a robust Single-Class Dataset for YOLO Stream B:
Target Class: 'bicycle' (Class 0).
Strictly Excluded: 'person', 'car', and all other classes.

Extracts positive bicycle images and negative samples (containing pedestrians/cars
with empty labels) so YOLO learns to detect ONLY bicycles and ignore cars & people.
"""

import os
import sys
import shutil
import random
from pathlib import Path


def prepare_single_class_bicycle_dataset(
    output_dir: Path,
    num_train_pos: int = 150,
    num_val_pos: int = 50,
    num_train_neg: int = 40,
    num_val_neg: int = 15,
    random_seed: int = 42
):
    random.seed(random_seed)

    # Directories
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    voc_labels_root = Path("C:/datasets/VOC/labels")
    voc_images_root = Path("C:/datasets/VOC/images")

    # In VOC: class 1 is 'bicycle', class 6 is 'car', class 14 is 'person'
    BICYCLE_VOC_ID = 1

    # Find all bicycle images across VOC splits
    bicycle_samples = []
    negative_samples = []

    # Map image stem to image file path
    image_file_map = {}
    for img_path in voc_images_root.rglob("*.jpg"):
        image_file_map[img_path.stem] = img_path

    label_files = list(voc_labels_root.rglob("*.txt"))
    random.shuffle(label_files)

    for lbl_file in label_files:
        stem = lbl_file.stem
        if stem not in image_file_map:
            continue
        img_path = image_file_map[stem]

        target_boxes = []
        has_car_or_person = False

        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id == BICYCLE_VOC_ID:
                        # Remap bicycle to class 0
                        target_boxes.append(f"0 {' '.join(parts[1:])}\n")
                    elif cls_id in (6, 14): # car or person
                        has_car_or_person = True

        if len(target_boxes) > 0:
            bicycle_samples.append((img_path, target_boxes))
        elif has_car_or_person:
            # High-value negative sample: image contains cars or pedestrians,
            # but label will be empty so model learns they are NOT bicycles!
            negative_samples.append((img_path, []))

    print(f"Total available bicycle samples found: {len(bicycle_samples)}")
    print(f"Total available negative samples (cars/people) found: {len(negative_samples)}")

    # Sample required amounts
    selected_pos_train = bicycle_samples[:num_train_pos]
    selected_pos_val = bicycle_samples[num_train_pos : num_train_pos + num_val_pos]

    selected_neg_train = negative_samples[:num_train_neg]
    selected_neg_val = negative_samples[num_train_neg : num_train_neg + num_val_neg]

    # Also extract background samples directly from CCTV surveillance video
    # (teaches the model that surveillance road/pavement/cars/pedestrians are negative background)
    cctv_video_path = Path("CCTV 01/VIRAT_S_010001_00_000019_000065.mp4")
    cctv_negatives = []
    if cctv_video_path.exists():
        import cv2
        cap = cv2.VideoCapture(str(cctv_video_path))
        cctv_temp_dir = output_dir / "cctv_bg_temp"
        cctv_temp_dir.mkdir(parents=True, exist_ok=True)
        frame_indices = [5, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400]
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                bg_img_path = cctv_temp_dir / f"cctv_bg_frame_{f_idx:04d}.jpg"
                cv2.imwrite(str(bg_img_path), frame)
                cctv_negatives.append((bg_img_path, []))
        cap.release()
        print(f"Extracted {len(cctv_negatives)} surveillance background negative frames from CCTV video.")

    # Split CCTV negatives: 10 train, 5 val
    cctv_train_neg = cctv_negatives[:10]
    cctv_val_neg = cctv_negatives[10:]

    train_data = selected_pos_train + selected_neg_train + cctv_train_neg
    val_data = selected_pos_val + selected_neg_val + cctv_val_neg
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"Selected Train: {len(selected_pos_train)} bicycles + {len(selected_neg_train) + len(cctv_train_neg)} negatives = {len(train_data)} total")
    print(f"Selected Val:   {len(selected_pos_val)} bicycles + {len(selected_neg_val) + len(cctv_val_neg)} negatives = {len(val_data)} total")

    # Copy files
    def copy_samples(samples, img_dest, lbl_dest):
        for img_path, boxes in samples:
            shutil.copy2(img_path, img_dest / img_path.name)
            lbl_dest_file = lbl_dest / f"{img_path.stem}.txt"
            with open(lbl_dest_file, "w", encoding="utf-8") as f:
                f.writelines(boxes)

    copy_samples(train_data, train_img_dir, train_lbl_dir)
    copy_samples(val_data, val_img_dir, val_lbl_dir)

    if (output_dir / "cctv_bg_temp").exists():
        shutil.rmtree(output_dir / "cctv_bg_temp")

    # Generate data.yaml
    yaml_content = f"""# Single-Class Known Object Dataset: bicycle
# Automatically curated for Stream B Known-Object Detector
path: {output_dir.as_posix()}
train: images/train
val: images/val
nc: 1
names:
  0: bicycle
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[OK] Single-class dataset successfully created at: {output_dir}")
    print(f"[OK] data.yaml created at: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent
    final_dataset_dir = workspace_root / "datasets" / "single_object_dataset"

    yaml_path = prepare_single_class_bicycle_dataset(final_dataset_dir)
