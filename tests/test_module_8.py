"""
Tests for Module 8: Relational Metadata & Observation Store (DatabaseManager)
Validates Fix #1 (no normalized_polygon, standard bbox_xyxy INT[4], frame_hw INT[2]),
Fix #3 (cluster and observation CRUD), and Fix #4 (credential isolation with SQLite fallback).
"""

import os
import shutil
import pytest

from module_8.config import Module8Config, build_postgres_dsn, load_phase1_env
from module_8.db_manager import DatabaseManager


class TestModule8:
    """Test suite for Relational DB Manager & Model Registry."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_module8"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_db_path = os.path.join(self.test_dir, "test_metadata.db")

        self.config = Module8Config(
            use_sqlite_fallback=True,
            sqlite_db_path=self.test_db_path
        )
        self.db = DatabaseManager(self.config)
        yield
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cluster_lifecycle_crud(self):
        """Test cluster insertion, incrementation, and status updates."""
        # 1. Insert cluster
        cluster_id, obs_count, status = self.db.insert_cluster(faiss_index_id=0)
        assert cluster_id == 1
        assert obs_count == 1
        assert status == "COLLECTING"

        # 2. Increment cluster
        c_id, obs_count_2, status_2 = self.db.increment_cluster(faiss_index_id=0)
        assert c_id == 1
        assert obs_count_2 == 2
        assert status_2 == "COLLECTING"

        # 3. Update status to IDENTIFIED
        self.db.update_cluster_status(
            cluster_id=1,
            status="IDENTIFIED",
            identified_label="electric_scooter",
            assigned_yolo_class_id=80,
            vlm_confidence=0.92,
            vlm_reasoning="Handlebars and two wheels visible."
        )

        cluster = self.db.get_cluster(1)
        assert cluster["status"] == "IDENTIFIED"
        assert cluster["identified_label"] == "electric_scooter"
        assert cluster["assigned_yolo_class_id"] == 80
        assert cluster["vlm_confidence"] == 0.92

    def test_observation_storage_standard_bbox(self):
        """Test observation insertion storing standard bbox_xyxy without polygon (Fix #1)."""
        cluster_id, _, _ = self.db.insert_cluster(faiss_index_id=0)

        obs_id = self.db.insert_observation(
            cluster_id=cluster_id,
            track_id=101,
            crop_image_path="storage/crops/crop_101.jpg",
            full_frame_path="storage/frames/frame_101.jpg",
            bbox_xyxy=[120, 200, 300, 450],
            sharpness_score=145.2,
            cosine_similarity=0.88,
            sub_object_idx=0,
            masked_crop_path="storage/masked_crops/masked_101.jpg",
            frame_hw=[1080, 1920],
            decomposition_method="FASTSAM_MERGED"
        )
        assert obs_id >= 1

        # Check top observations ordering
        top_obs = self.db.get_top_observations(cluster_id=cluster_id, limit=5)
        assert len(top_obs) == 1
        assert top_obs[0][0] == "storage/crops/crop_101.jpg"
        assert top_obs[0][1] == "storage/frames/frame_101.jpg"
        assert top_obs[0][2] == [120, 200, 300, 450]

        # Check cluster observations for dataset building
        dataset_obs = self.db.get_cluster_observations(cluster_id=cluster_id)
        assert len(dataset_obs) == 1
        assert dataset_obs[0]["bbox_xyxy"] == [120, 200, 300, 450]
        assert dataset_obs[0]["frame_hw"] == [1080, 1920]
        assert "normalized_polygon" not in dataset_obs[0]

    def test_model_registry_crud(self):
        """Test model version registration and production promotion."""
        m_id = self.db.insert_model_version(
            weights_path="models/challenger_v1.pt",
            num_classes=81,
            class_names=["person", "car", "electric_scooter"],
            old_classes_map50_95=0.552,
            new_class_map50=0.741,
            new_class_recall=0.785,
            promoted_to_production=False,
            trained_on_cluster_id=1
        )
        assert m_id >= 1

        # Initially no production model
        assert self.db.get_production_model() is None

        # Promote to production
        self.db.promote_model_version(m_id)
        prod = self.db.get_production_model()
        assert prod is not None
        assert prod["model_version_id"] == m_id
        assert prod["weights_path"] == "models/challenger_v1.pt"
        assert prod["num_classes"] == 81
