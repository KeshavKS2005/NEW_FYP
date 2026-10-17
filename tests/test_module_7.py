"""
Tests for Module 7: Online FAISS Centroid Clustering & Relational DB Persistence
"""

import os
import shutil
import pytest
import numpy as np

from module_7.config import Module7Config
from module_7.db_manager import DatabaseManager
from module_7.faiss_clustering import FAISSClusterEngine


class TestModule7:
    """Test suite for FAISS cosine centroid matching and database operations."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_module7"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_db_path = os.path.join(self.test_dir, "test_discovery.db")
        self.test_index_path = os.path.join(self.test_dir, "test_faiss.index")
        self.test_centroids_path = os.path.join(self.test_dir, "test_centroids.npy")

        self.config = Module7Config(
            embedding_dim=384,
            cosine_threshold=0.78,
            use_sqlite_fallback=True,
            sqlite_db_path=self.test_db_path,
            index_path=self.test_index_path,
            centroids_path=self.test_centroids_path
        )
        self.db = DatabaseManager(self.config)
        self.engine = FAISSClusterEngine(self.config, db_manager=self.db)
        yield
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _random_l2_vector(self) -> np.ndarray:
        vec = np.random.randn(1, 384).astype("float32")
        return vec / np.linalg.norm(vec)

    def test_first_cluster_instantiation(self):
        """First observation must create cluster 1 with observation_count = 1 and sim = 1.0."""
        vec1 = self._random_l2_vector()
        c_id, obs_count, status, sim = self.engine.add_or_cluster(
            embedding=vec1,
            track_id=101,
            frame_path="storage/frames/t101.jpg",
            crop_path="storage/crops/t101.jpg",
            bbox=(10, 20, 50, 60),
            sharpness=120.5
        )

        assert c_id == 1
        assert obs_count == 1
        assert status == "COLLECTING"
        assert sim == 1.0
        assert self.engine.total_clusters == 1

        # Check DB
        cluster_info = self.db.get_cluster(1)
        assert cluster_info["observation_count"] == 1

    def test_similar_vector_merges_and_increments(self):
        """Vectors with cosine similarity >= 0.78 must merge into existing cluster."""
        vec1 = self._random_l2_vector()
        self.engine.add_or_cluster(
            embedding=vec1,
            track_id=1,
            frame_path="f1.jpg",
            crop_path="c1.jpg",
            bbox=(0, 0, 10, 10),
            sharpness=100.0
        )

        # Create nearly identical vector (cosine sim ~ 0.99)
        noise = np.random.randn(1, 384).astype("float32")
        noise = noise / np.linalg.norm(noise)
        vec2 = vec1 + 0.1 * noise
        vec2 = vec2 / np.linalg.norm(vec2)

        c_id, obs_count, status, sim = self.engine.add_or_cluster(
            embedding=vec2,
            track_id=2,
            frame_path="f2.jpg",
            crop_path="c2.jpg",
            bbox=(5, 5, 15, 15),
            sharpness=110.0
        )

        assert c_id == 1, f"Expected cluster 1, got {c_id}"
        assert obs_count == 2
        assert sim >= 0.78
        assert self.engine.total_clusters == 1

        # Check top observations
        top_obs = self.db.get_top_observations(1, limit=3)
        assert len(top_obs) == 2
        # Highest sharpness (110.0) should be first
        assert top_obs[0][0] == "c2.jpg"
        assert top_obs[0][1] == "f2.jpg"
        assert top_obs[0][2] == [5, 5, 15, 15]

    def test_dissimilar_vector_creates_new_cluster(self):
        """Orthogonal/dissimilar vector (cosine sim < 0.78) must create a new cluster."""
        vec1 = np.zeros((1, 384), dtype="float32")
        vec1[0, 0] = 1.0  # Unit vector on axis 0

        self.engine.add_or_cluster(
            embedding=vec1,
            track_id=1,
            frame_path="f1.jpg",
            crop_path="c1.jpg",
            bbox=(0, 0, 10, 10),
            sharpness=100.0
        )

        vec2 = np.zeros((1, 384), dtype="float32")
        vec2[0, 1] = 1.0  # Unit vector on axis 1 (orthogonal, dot product = 0.0)

        c_id, obs_count, status, sim = self.engine.add_or_cluster(
            embedding=vec2,
            track_id=2,
            frame_path="f2.jpg",
            crop_path="c2.jpg",
            bbox=(10, 10, 20, 20),
            sharpness=90.0
        )

        assert c_id == 2, f"Expected novel cluster 2, got {c_id}"
        assert obs_count == 1
        assert self.engine.total_clusters == 2

    def test_faiss_persistence_and_reload_sync(self):
        """FAISS index and centroids must persist to disk and reload on engine re-instantiation (Fix #3)."""
        vec1 = self._random_l2_vector()
        self.engine.add_or_cluster(
            embedding=vec1,
            track_id=1,
            frame_path="f1.jpg",
            crop_path="c1.jpg",
            bbox=(0, 0, 10, 10),
            sharpness=100.0
        )
        assert os.path.exists(self.test_index_path)
        assert os.path.exists(self.test_centroids_path)

        # Create a new engine with the same config - must restore previous cluster
        new_engine = FAISSClusterEngine(self.config, db_manager=self.db)
        assert new_engine.total_clusters == 1

        # Query with same vector should match existing cluster
        c_id, obs_count, status, sim = new_engine.add_or_cluster(
            embedding=vec1,
            track_id=2,
            frame_path="f2.jpg",
            crop_path="c2.jpg",
            bbox=(5, 5, 15, 15),
            sharpness=105.0
        )
        assert c_id == 1
        assert obs_count == 2
        assert sim >= 0.99
