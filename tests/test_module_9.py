"""
Tests for Module 9: Automated VLM Discovery Trigger (Gemini 2.5 Flash)
Validates Fix #2 (Clean Full Frames for YOLO Retraining vs. Red-Box Frames for Gemini),
Pydantic structured output schema, threshold gating, and retrain_queue dispatch.
"""

import os
import shutil
import pytest
import numpy as np
import cv2
from multiprocessing import Queue

from module_8.config import Module8Config
from module_8.db_manager import DatabaseManager
from module_9.config import Module9Config
from module_9.vlm_identifier import GeminiObjectSchema, VLMDiscoveryIdentifier


class TestModule9:
    """Test suite for VLM Discovery Trigger, clean frame preservation, and Gemini dispatch."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_module9"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_db_path = os.path.join(self.test_dir, "test_vlm.db")

        self.m8_cfg = Module8Config(
            use_sqlite_fallback=True,
            sqlite_db_path=self.test_db_path
        )
        self.db = DatabaseManager(self.m8_cfg)
        self.config = Module9Config(trigger_count=5, min_confidence_for_retrain=0.70)
        self.identifier = VLMDiscoveryIdentifier(self.config)
        yield
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pydantic_schema_validation(self):
        """GeminiObjectSchema must enforce strict JSON structure."""
        valid_json = """{
            "object_name": "electric_scooter",
            "category": "vehicle",
            "is_valid_physical_object": true,
            "confidence": 0.95,
            "reasoning": "Visible handlebars, two wheels, moving along pedestrian curb."
        }"""
        obj = GeminiObjectSchema.model_validate_json(valid_json)
        assert obj.object_name == "electric_scooter"
        assert obj.category == "vehicle"
        assert obj.is_valid_physical_object is True
        assert obj.confidence == 0.95

    def test_trigger_threshold_condition(self):
        """VLM must only trigger when observation_count >= trigger_count and status is COLLECTING."""
        assert self.identifier.should_trigger(observation_count=4, status="COLLECTING") is False
        assert self.identifier.should_trigger(observation_count=5, status="COLLECTING") is True
        assert self.identifier.should_trigger(observation_count=100, status="COLLECTING") is True
        assert self.identifier.should_trigger(observation_count=100, status="IDENTIFIED") is False
        assert self.identifier.should_trigger(observation_count=100, status="DEPLOYED_IN_YOLO") is False

    def test_clean_full_frame_preservation_fix_2(self):
        """
        Fix #2 Verification: The pristine full_frame_path on disk MUST NOT be modified
        or have red bounding boxes permanently drawn on it when passed to Gemini.
        """
        c_id, obs_count, status = self.db.insert_cluster(faiss_index_id=0)

        # Create a pristine test frame on disk (all pixels pure blue: BGR [255, 0, 0])
        clean_frame_path = os.path.join(self.test_dir, "pristine_frame.jpg")
        crop_path = os.path.join(self.test_dir, "pristine_crop.jpg")

        pristine_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pristine_frame[:, :] = [255, 0, 0]  # pure blue
        cv2.imwrite(clean_frame_path, pristine_frame)

        pristine_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        pristine_crop[:, :] = [200, 200, 200]
        cv2.imwrite(crop_path, pristine_crop)

        # Record observation with bbox [50, 50, 150, 150]
        bbox = [50, 50, 150, 150]
        self.db.insert_observation(
            cluster_id=c_id,
            track_id=1,
            crop_image_path=crop_path,
            full_frame_path=clean_frame_path,
            bbox_xyxy=bbox,
            sharpness_score=180.0,
            cosine_similarity=1.0
        )

        retrain_q = Queue()

        # Run identification
        res = self.identifier.identify_cluster(
            cluster_id=c_id,
            db=self.db,
            retrain_queue=retrain_q
        )

        assert res is not None
        assert res.is_valid_physical_object is True

        # CRUCIAL CHECK: Read back clean_frame_path from disk. It must NOT contain red pixels!
        frame_after = cv2.imread(clean_frame_path)
        assert frame_after is not None
        # In BGR, a red box drawn by cv2 has BGR [0, 0, 255]. Check that no pixels on disk are pure red:
        red_pixels = np.all(frame_after == [0, 0, 255], axis=-1)
        assert not np.any(red_pixels), "Error: Red box was permanently written to clean_frame_path on disk!"

        # Verify DB updated
        cluster_info = self.db.get_cluster(c_id)
        assert cluster_info["status"] == "IDENTIFIED"
        assert cluster_info["identified_label"] == res.object_name
        assert cluster_info["vlm_confidence"] == res.confidence

        # Verify task dispatched to retrain_queue
        assert not retrain_q.empty()
        item = retrain_q.get_nowait()
        assert item == c_id
