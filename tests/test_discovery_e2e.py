"""
End-to-End Discovery Pipeline Integration Tests (Modules 1-4 -> Worker -> Modules 5-8)
"""

import os
import shutil
import pytest
import numpy as np
import cv2

from module_4.config import Module4Config
from module_4.temporal_gater import TemporalTrajectoryGater
from module_5.config import Module5Config
from module_6.config import Module6Config
from module_7.config import Module7Config
from module_8.config import Module8Config
from module_8.db_manager import DatabaseManager
from module_9.config import Module9Config
from discovery_worker import DiscoveryWorker


class TestDiscoveryE2E:
    """Full lifecycle integration tests for discovery pipeline."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_e2e"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "e2e_discovery.db")

        self.m5_cfg = Module5Config()
        self.m6_cfg = Module6Config()
        self.m7_cfg = Module7Config(
            cosine_threshold=0.78,
            index_path=os.path.join(self.test_dir, "test_faiss.index"),
            centroids_path=os.path.join(self.test_dir, "test_centroids.npy")
        )
        self.m8_cfg = Module8Config(
            use_sqlite_fallback=True,
            sqlite_db_path=self.db_path
        )
        self.m9_cfg = Module9Config(
            trigger_count=2,
            allow_mock_fallback=True
        )

        self.db = DatabaseManager(self.m8_cfg)
        self.worker = DiscoveryWorker(
            m5_config=self.m5_cfg,
            m6_config=self.m6_cfg,
            m7_config=self.m7_cfg,
            m8_config=self.m8_cfg,
            m9_config=self.m9_cfg,
            db_manager=self.db
        )
        yield
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_worker_process_item_flow(self):
        """Worker must process keyframe item, extract DINOv2 embedding, cluster, and store in DB."""
        # 1. Create synthetic frame and crop
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw_crop = np.full((80, 80, 3), 180, dtype=np.uint8)
        raw_crop[::5, :] = 255  # Add texture for sharpness
        mask_crop = np.full((80, 80), 255, dtype=np.uint8)
        bbox = (100, 100, 180, 180)
        sharpness = 125.0

        item = (1, frame, raw_crop, mask_crop, bbox, sharpness)

        # 2. Process first item
        res1 = self.worker.process_item(item)
        assert res1["track_id"] == 1
        assert res1["cluster_id"] == 1
        assert res1["observation_count"] == 1
        assert res1["status"] == "COLLECTING"
        assert os.path.exists(res1["frame_path"])
        assert os.path.exists(res1["crop_path"])

        # 3. Process second item of same object -> hits trigger_count = 2!
        item2 = (2, frame, raw_crop, mask_crop, bbox, sharpness + 10.0)
        res2 = self.worker.process_item(item2)
        assert res2["cluster_id"] == 1
        assert res2["observation_count"] == 2
        assert res2["similarity"] >= 0.78
        assert res2["status"] == "COLLECTING"  # Status before trigger
        assert res2["vlm_result"] is not None
        assert res2["vlm_result"].is_valid_physical_object is True

        # Check final DB state
        cluster_record = self.db.get_cluster(1)
        assert cluster_record["status"] == "IDENTIFIED"
        assert cluster_record["observation_count"] == 2
        assert cluster_record["identified_label"] == "mock_discovered_object"

    def test_module4_to_worker_emission_flow(self):
        """Module 4 must buffer sharpest frames and emit completed track payload on track exit."""
        gater = TemporalTrajectoryGater(Module4Config(
            survival_window=5,
            active_learning=False,
            max_lost_frames=5
        ))

        # Simulate object moving linearly across frames (x increases by 10 each frame)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        fg_mask = np.zeros((300, 300), dtype=np.uint8)

        emitted_payloads = []
        # Run 8 frames of consistent motion
        for f in range(8):
            x = 20 + f * 10
            y = 50
            box = np.array([[x, y, x + 30, y + 30, 0.95]], dtype=np.float32)
            # Add synthetic motion inside box
            fg_mask[y:y+30, x:x+30] = 255
            frame[y:y+30, x:x+30] = (f * 25) % 255

            res = gater.update(
                candidate_boxes=box,
                frame=frame,
                frame_index=f + 1,
                fg_mask=fg_mask
            )
            if res.completed_track_payloads:
                emitted_payloads.extend(res.completed_track_payloads)

        # Track is confirmed and moving. Now simulate 6 empty frames (track lost & removed)
        empty_box = np.empty((0, 5), dtype=np.float32)
        for f in range(8, 16):
            res = gater.update(
                candidate_boxes=empty_box,
                frame=frame,
                frame_index=f + 1,
                fg_mask=fg_mask
            )
            if res.completed_track_payloads:
                emitted_payloads.extend(res.completed_track_payloads)

        assert len(emitted_payloads) == 1, (
            f"Expected exactly 1 completed track payload emitted, got {len(emitted_payloads)}"
        )

        completed_item = emitted_payloads[0]
        # completed_item is (track_id, frame_bgr, raw_crop, mask_crop, bbox, sharpness)
        assert len(completed_item) == 6
        assert completed_item[0] > 0  # track_id
        assert completed_item[1].shape == (300, 300, 3)
        assert completed_item[4] is not None  # bbox

        # Pass emitted payload to DiscoveryWorker
        worker_res = self.worker.process_item(completed_item)
        assert worker_res["cluster_id"] == 1
        assert worker_res["observation_count"] == 1
