"""
Tests for Module 10: Continual Dataset Builder & Champion Teacher Pseudo-Labeler
Validates Fix #1 (Standard 5-column YOLO bbox format only, no polygons),
Fix #2 (Pristine clean full frames used for training), and Experience Replay Buffering.
"""

import os
import shutil
from pathlib import Path
import pytest
import numpy as np
import cv2
import yaml

from module_8.config import Module8Config
from module_8.db_manager import DatabaseManager
from module_10.config import Module10Config
from module_10.dataset_builder import ContinualDatasetBuilder


class TestModule10:
    """Test suite for autonomous continual dataset creation and experience replay."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_module10"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_db_path = os.path.join(self.test_dir, "test_dataset.db")
        self.datasets_dir = os.path.join(self.test_dir, "datasets")
        self.replay_dir = os.path.join(self.test_dir, "replay_buffer")

        self.db_config = Module8Config(
            use_sqlite_fallback=True,
            sqlite_db_path=self.test_db_path
        )
        self.db = DatabaseManager(self.db_config)
        self.config = Module10Config(
            datasets_base_dir=self.datasets_dir,
            replay_buffer_dir=self.replay_dir,
            teacher_conf_threshold=0.50,
            train_split_ratio=0.75,
            replay_archive_count=5
        )
        self.builder = ContinualDatasetBuilder(self.config)
        yield
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dataset_generation_standard_5col_bbox(self):
        """
        Fix #1 & #2 Verification:
        1. Output labels must be standard 5-column format: <cls> <xc> <yc> <w> <h>.
        2. Full frames in dataset images must be pristine (no drawn red boxes).
        """
        # 1. Setup cluster in DB
        cluster_id, _, _ = self.db.insert_cluster(faiss_index_id=0)
        self.db.update_cluster_status(
            cluster_id=cluster_id,
            status="IDENTIFIED",
            identified_label="electric_scooter",
            assigned_yolo_class_id=80
        )

        # 2. Create 4 pristine test frames on disk
        frames_dir = os.path.join(self.test_dir, "raw_frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i in range(4):
            clean_frame = np.full((480, 640, 3), 120 + i * 10, dtype=np.uint8)
            fpath = os.path.join(frames_dir, f"frame_{i}.jpg")
            cpath = os.path.join(frames_dir, f"crop_{i}.jpg")
            cv2.imwrite(fpath, clean_frame)
            cv2.imwrite(cpath, clean_frame[50:150, 50:150])

            self.db.insert_observation(
                cluster_id=cluster_id,
                track_id=100 + i,
                crop_image_path=cpath,
                full_frame_path=fpath,
                bbox_xyxy=[64, 48, 192, 144],  # xc=128/640=0.2, yc=96/480=0.2, w=128/640=0.2, h=96/480=0.2
                sharpness_score=150.0 + i,
                cosine_similarity=0.90,
                frame_hw=[480, 640]
            )

        # 3. Build dataset using champion model yolo11n.pt
        ds_yaml_path, old_class_ids, target_cls_id, new_label = self.builder.build_dataset(
            cluster_id=cluster_id,
            champion_weights_path="yolo11n.pt",
            db=self.db
        )

        assert ds_yaml_path.exists()
        assert target_cls_id == 80
        assert new_label == "electric_scooter"
        assert len(old_class_ids) == 80

        # Verify dataset YAML configuration
        with open(ds_yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        assert "names" in yaml_data
        assert yaml_data["names"][80] == "electric_scooter"

        # Check label files: MUST BE STANDARD 5 COLUMNS ONLY
        ds_root = ds_yaml_path.parent
        label_files = list(ds_root.glob("labels/*/*.txt"))
        assert len(label_files) == 4

        for lf in label_files:
            content = lf.read_text().strip()
            assert len(content) > 0
            lines = content.splitlines()
            for line in lines:
                parts = line.strip().split()
                # Strict check for standard 5 columns (Fix #1)
                assert len(parts) == 5, f"Expected exactly 5 columns in YOLO bbox format, got {len(parts)}: {line}"
                cls_id = int(parts[0])
                xc, yc, w, h = [float(p) for p in parts[1:]]
                assert 0.0 <= xc <= 1.0
                assert 0.0 <= yc <= 1.0
                assert 0.0 < w <= 1.0
                assert 0.0 < h <= 1.0

        # Check images: MUST BE PRISTINE (Fix #2)
        image_files = list(ds_root.glob("images/*/*.jpg"))
        assert len(image_files) == 4
        for imf in image_files:
            img = cv2.imread(str(imf))
            assert img is not None
            # Verify no red bounding box pixels (BGR [0, 0, 255])
            red_pixels = np.all(img == [0, 0, 255], axis=-1)
            assert not np.any(red_pixels), f"Found red annotation on training frame {imf}!"

        # Check Experience Replay Buffer archival
        replay_imgs = list((Path(self.replay_dir) / "images").glob("*.jpg"))
        replay_lbls = list((Path(self.replay_dir) / "labels").glob("*.txt"))
        assert len(replay_imgs) == 4
        assert len(replay_lbls) == 4
