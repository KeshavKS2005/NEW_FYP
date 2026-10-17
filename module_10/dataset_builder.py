"""
Module 10: Continual Dataset Builder & Teacher Pseudo-Labeler
Prepares a self-contained Ultralytics YOLO standard bounding-box dataset whenever a new cluster is identified.
Prevents Unlabeled Foreground Suppression via Teacher Pseudo-Labeling and Catastrophic Forgetting via Experience Replay.
Adheres strictly to Module 10 in fyp_architecture_final_object.md.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Any
import numpy as np
import yaml
from ultralytics import YOLO

from module_10.config import Module10Config
from module_8.db_manager import DatabaseManager


class ContinualDatasetBuilder:
    """
    Builds incremental YOLO datasets with:
    1. Teacher pseudo-labeling on clean full frames with active Champion model.
    2. Novel class standard 5-column bounding box annotations.
    3. Experience Replay Buffer injection and archival.
    """

    def __init__(self, config: Optional[Module10Config] = None):
        self.config = config or Module10Config()
        self.replay_img_dir = Path(self.config.replay_buffer_dir) / "images"
        self.replay_lbl_dir = Path(self.config.replay_buffer_dir) / "labels"
        self.replay_img_dir.mkdir(parents=True, exist_ok=True)
        self.replay_lbl_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(
        self,
        cluster_id: int,
        champion_weights_path: str,
        db: DatabaseManager
    ) -> Tuple[Path, List[int], int, str]:
        """
        Builds the YOLO dataset for a newly identified cluster.

        Args:
            cluster_id: ID of the identified cluster.
            champion_weights_path: Path to active production Champion YOLO weights.
            db: Relational database manager instance.

        Returns:
            Tuple of (dataset_yaml_path, old_class_ids, target_cls_id, new_label_name).
        """
        cluster_info = db.get_cluster(cluster_id)
        if not cluster_info or not cluster_info.get("identified_label"):
            raise ValueError(f"Cluster {cluster_id} is not identified or has no label.")

        new_label_name = cluster_info["identified_label"].strip().lower()

        # Load Champion model to obtain existing class catalog
        champion_model = YOLO(champion_weights_path)
        existing_names = dict(champion_model.names) if hasattr(champion_model, "names") and champion_model.names else {}
        name_to_id = {v.lower(): k for k, v in existing_names.items()}

        if new_label_name in name_to_id:
            target_cls_id = name_to_id[new_label_name]
        else:
            target_cls_id = max(existing_names.keys()) + 1 if existing_names else 0
            existing_names[target_cls_id] = new_label_name

        old_class_ids = [cid for cid in existing_names.keys() if cid != target_cls_id]

        ds_root = Path(self.config.datasets_base_dir) / f"cluster_{cluster_id}"
        for split in ("train", "val"):
            (ds_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (ds_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        obs_rows = db.get_cluster_observations(cluster_id)
        if not obs_rows:
            raise ValueError(f"No observations found for cluster {cluster_id}.")

        split_idx = max(1, int(self.config.train_split_ratio * len(obs_rows)))

        for idx, obs in enumerate(obs_rows):
            obs_id = obs["observation_id"]
            frame_path = obs["full_frame_path"]
            bbox = obs["bbox_xyxy"]
            frame_hw = obs["frame_hw"]

            if not os.path.exists(frame_path):
                continue

            split = "train" if idx < split_idx else "val"
            dst_img = ds_root / "images" / split / f"obs_{obs_id}.jpg"
            dst_lbl = ds_root / "labels" / split / f"obs_{obs_id}.txt"
            shutil.copy(frame_path, dst_img)

            label_lines: List[str] = []
            fh, fw = frame_hw

            # 10.1 Teacher Pseudo-Labeling: Annotate existing known objects on clean frame
            try:
                teacher_res = champion_model.predict(
                    source=frame_path,
                    conf=self.config.teacher_conf_threshold,
                    verbose=False
                )[0]
                if teacher_res.boxes is not None and len(teacher_res.boxes) > 0:
                    t_boxes = teacher_res.boxes.xyxy.cpu().numpy()
                    t_clses = teacher_res.boxes.cls.cpu().numpy().astype(int)
                    for (tx1, ty1, tx2, ty2), t_cls in zip(t_boxes, t_clses):
                        if t_cls in self.config.holdout_class_ids:
                            continue
                        xc = np.clip(((tx1 + tx2) / 2.0) / fw, 0.0, 1.0)
                        yc = np.clip(((ty1 + ty2) / 2.0) / fh, 0.0, 1.0)
                        w = np.clip((tx2 - tx1) / fw, 0.0, 1.0)
                        h = np.clip((ty2 - ty1) / fh, 0.0, 1.0)
                        label_lines.append(f"{t_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            except Exception as e:
                print(f"[Module 10] Teacher pseudo-labeling warning on frame {frame_path}: {e}")

            # 10.2 Append newly discovered object's standard 5-column YOLO bounding box
            bx1, by1, bx2, by2 = bbox
            xc = np.clip(((bx1 + bx2) / 2.0) / fw, 0.0, 1.0)
            yc = np.clip(((by1 + by2) / 2.0) / fh, 0.0, 1.0)
            w = np.clip((bx2 - bx1) / fw, 0.0, 1.0)
            h = np.clip((by2 - by1) / fh, 0.0, 1.0)
            label_lines.append(f"{target_cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            dst_lbl.write_text("\n".join(label_lines) + "\n")

            # 10.3 Archive top sharpest samples into Experience Replay Buffer for future cycles
            if idx < self.config.replay_archive_count:
                shutil.copy(dst_img, self.replay_img_dir / f"cluster_{cluster_id}_obs_{obs_id}.jpg")
                shutil.copy(dst_lbl, self.replay_lbl_dir / f"cluster_{cluster_id}_obs_{obs_id}.txt")

        # 10.4 Inject existing Experience Replay Buffer samples into Train & Val splits
        replay_imgs = sorted(list(self.replay_img_dir.glob("*.jpg")))
        for r_idx, r_img in enumerate(replay_imgs):
            # Don't duplicate current cluster samples
            if f"cluster_{cluster_id}_" in r_img.name:
                continue
            r_lbl = self.replay_lbl_dir / f"{r_img.stem}.txt"
            if r_lbl.exists():
                r_split = "train" if (r_idx % 5 != 0) else "val"
                shutil.copy(r_img, ds_root / "images" / r_split / r_img.name)
                shutil.copy(r_lbl, ds_root / "labels" / r_split / r_lbl.name)

        dataset_yaml_path = ds_root / "dataset.yaml"
        dataset_config = {
            "path": str(ds_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": existing_names,
        }
        dataset_yaml_path.write_text(yaml.dump(dataset_config, sort_keys=True))

        return dataset_yaml_path, old_class_ids, target_cls_id, new_label_name


AutonomousDatasetBuilder = ContinualDatasetBuilder
