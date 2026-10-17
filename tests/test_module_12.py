"""
Tests for Module 12: Atomic Model Hot-Swapper & Dynamic YOLO Reloader
Validates atomic staging & replacement (os.replace), model registry logging,
cluster status transitions, and hot-swap event signaling.
"""

import os
import shutil
from pathlib import Path
from multiprocessing import Event
import pytest

from module_8.config import Module8Config
from module_8.db_manager import DatabaseManager
from module_12.config import Module12Config
from module_12.hot_swapper import ModelHotSwapper


class TestModule12:
    """Test suite for atomic hot-swap promotion and rollback."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_module12"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_db_path = os.path.join(self.test_dir, "test_hotswap.db")

        self.db_config = Module8Config(
            use_sqlite_fallback=True,
            sqlite_db_path=self.test_db_path
        )
        self.db = DatabaseManager(self.db_config)

        self.champ_path = os.path.join(self.test_dir, "champion_active.pt")
        self.staging_path = os.path.join(self.test_dir, "champion_staging.pt")

        # Create active champion weights file with initial content
        with open(self.champ_path, "wb") as f:
            f.write(b"CHAMPION_V0_WEIGHTS")

        self.config = Module12Config(
            champion_weights_path=self.champ_path,
            staging_weights_path=self.staging_path
        )
        self.swapper = ModelHotSwapper(self.config)
        yield
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_promotion_atomic_swap_and_event(self):
        """Promote=True must atomically replace weights, update DB, and fire reload event."""
        # Setup cluster
        c_id, _, _ = self.db.insert_cluster(faiss_index_id=0)
        self.db.update_cluster_status(c_id, "RETRAINING", identified_label="electric_scooter")

        # Create challenger file
        chal_weights = Path(self.test_dir) / "challenger_best.pt"
        chal_weights.write_bytes(b"CHALLENGER_V1_WEIGHTS")

        metrics = {
            "chal_old_map": 0.525,
            "new_cls_map50": 0.780,
            "new_cls_recall": 0.810
        }
        reload_event = Event()

        m_id = self.swapper.deploy_or_reject(
            challenger_weights_path=chal_weights,
            promote=True,
            metrics=metrics,
            cluster_id=c_id,
            target_cls_id=80,
            db=self.db,
            model_reload_event=reload_event
        )

        assert m_id >= 1
        # 1. Champion file must now have new weights
        assert Path(self.champ_path).read_bytes() == b"CHALLENGER_V1_WEIGHTS"
        # 2. Reload event must be signaled
        assert reload_event.is_set() is True
        # 3. Cluster status must be DEPLOYED_IN_YOLO
        cluster = self.db.get_cluster(c_id)
        assert cluster["status"] == "DEPLOYED_IN_YOLO"
        assert cluster["assigned_yolo_class_id"] == 80

    def test_rejection_retains_champion_weights(self):
        """Promote=False must NOT modify active weights and must NOT fire reload event."""
        c_id, _, _ = self.db.insert_cluster(faiss_index_id=1)
        self.db.update_cluster_status(c_id, "RETRAINING", identified_label="bad_candidate")

        chal_weights = Path(self.test_dir) / "challenger_rejected.pt"
        chal_weights.write_bytes(b"REJECTED_WEIGHTS")

        metrics = {
            "chal_old_map": 0.400,  # Dropped too much
            "new_cls_map50": 0.500,
            "new_cls_recall": 0.500
        }
        reload_event = Event()

        m_id = self.swapper.deploy_or_reject(
            challenger_weights_path=chal_weights,
            promote=False,
            metrics=metrics,
            cluster_id=c_id,
            target_cls_id=81,
            db=self.db,
            model_reload_event=reload_event
        )

        assert m_id >= 1
        # 1. Champion weights must remain UNCHANGED
        assert Path(self.champ_path).read_bytes() == b"CHAMPION_V0_WEIGHTS"
        # 2. Reload event must NOT be signaled
        assert reload_event.is_set() is False
        # 3. Cluster status must be IDENTIFIED (awaiting more data/retraining)
        cluster = self.db.get_cluster(c_id)
        assert cluster["status"] == "IDENTIFIED"
