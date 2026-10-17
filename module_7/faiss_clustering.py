"""
Module 7: Online FAISS Centroid Clustering Engine
Performs online cosine similarity matching (threshold >= 0.78) using faiss.IndexFlatIP.
Updates running centroids and synchronizes FAISS index to disk across process restarts.
Adheres strictly to Module 7 in fyp_architecture_final_object.md.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Any
import faiss
import numpy as np

from module_7.config import Module7Config


class FAISSClusterEngine:
    """
    Online centroid clustering engine powered by FAISS Inner Product (Cosine Similarity).
    Synchronizes index and centroids to disk on update.
    """

    def __init__(
        self,
        config: Optional[Module7Config] = None,
        db_manager: Optional[Any] = None
    ):
        self.config = config or Module7Config()
        self.db = db_manager
        self.index = faiss.IndexFlatIP(self.config.embedding_dim)
        self.centroids: List[np.ndarray] = []

        # Load from disk if persisted index exists
        self.load_from_disk()

    def save_to_disk(self):
        """Persists FAISS index and centroids list to disk."""
        try:
            Path(self.config.index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, self.config.index_path)
            if self.centroids:
                np.save(self.config.centroids_path, np.array(self.centroids, dtype="float32"))
        except Exception as e:
            print(f"[Module 7] Warning: Failed to save FAISS index to disk: {e}")

    def load_from_disk(self) -> bool:
        """Loads FAISS index and centroids from disk if present."""
        if os.path.exists(self.config.index_path) and os.path.exists(self.config.centroids_path):
            try:
                self.index = faiss.read_index(self.config.index_path)
                loaded_c = np.load(self.config.centroids_path)
                self.centroids = [loaded_c[i] for i in range(len(loaded_c))]
                return True
            except Exception as e:
                print(f"[Module 7] Warning: Failed to load FAISS index from disk: {e}")
        return False

    def match_or_add(
        self,
        embedding: np.ndarray,
        observation_count: int = 1
    ) -> Tuple[int, float, bool]:
        """
        Pure FAISS centroid matching & running average update.

        Args:
            embedding: L2-normalized float32 numpy array (1, 384).
            observation_count: Current observation count of the cluster (for running average).

        Returns:
            Tuple of (faiss_index_id, cosine_similarity, is_new_cluster).
        """
        emb = embedding.astype("float32")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        if self.index.ntotal > 0:
            D, I = self.index.search(emb, 1)
            best_sim, best_faiss_id = float(D[0][0]), int(I[0][0])
        else:
            best_sim, best_faiss_id = -1.0, -1

        if best_sim >= self.config.cosine_threshold:
            # Running centroid update
            new_c = (observation_count - 1) * self.centroids[best_faiss_id] + emb[0]
            norm_c = np.linalg.norm(new_c)
            if norm_c > 1e-12:
                self.centroids[best_faiss_id] = (new_c / norm_c).astype("float32")
            else:
                self.centroids[best_faiss_id] = emb[0]

            self.index.reset()
            self.index.add(np.array(self.centroids, dtype="float32"))
            self.save_to_disk()
            return best_faiss_id, best_sim, False
        else:
            new_faiss_id = self.index.ntotal
            self.index.add(emb)
            self.centroids.append(emb[0].copy())
            self.save_to_disk()
            return new_faiss_id, 1.0, True

    def add_or_cluster(
        self,
        embedding: np.ndarray,
        track_id: int,
        frame_path: str,
        crop_path: str,
        bbox: Tuple[int, int, int, int],
        sharpness: float
    ) -> Tuple[int, int, str, float]:
        """
        Backward-compatible helper linking FAISS matching with database logging.
        """
        if self.db is None:
            from module_8.db_manager import DatabaseManager
            self.db = DatabaseManager()

        emb = embedding.astype("float32")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        if self.index.ntotal > 0:
            D, I = self.index.search(emb, 1)
            best_sim, best_faiss_id = float(D[0][0]), int(I[0][0])
        else:
            best_sim, best_faiss_id = -1.0, -1

        if best_sim >= self.config.cosine_threshold:
            cluster_id, obs_count, status = self.db.increment_cluster(best_faiss_id)
            new_c = (obs_count - 1) * self.centroids[best_faiss_id] + emb[0]
            norm_c = np.linalg.norm(new_c)
            if norm_c > 1e-12:
                self.centroids[best_faiss_id] = (new_c / norm_c).astype("float32")
            else:
                self.centroids[best_faiss_id] = emb[0]

            self.index.reset()
            self.index.add(np.array(self.centroids, dtype="float32"))
        else:
            new_faiss_id = self.index.ntotal
            self.index.add(emb)
            self.centroids.append(emb[0].copy())
            best_sim = 1.0
            cluster_id, obs_count, status = self.db.insert_cluster(new_faiss_id)

        self.save_to_disk()

        self.db.insert_observation(
            cluster_id=cluster_id,
            track_id=track_id,
            crop_image_path=crop_path,
            full_frame_path=frame_path,
            bbox_xyxy=list(bbox),
            sharpness_score=sharpness,
            cosine_similarity=best_sim
        )

        return cluster_id, obs_count, status, best_sim

    @property
    def total_clusters(self) -> int:
        return self.index.ntotal

    def reset(self):
        self.index.reset()
        self.centroids.clear()
        if os.path.exists(self.config.index_path):
            try:
                os.remove(self.config.index_path)
            except Exception:
                pass
        if os.path.exists(self.config.centroids_path):
            try:
                os.remove(self.config.centroids_path)
            except Exception:
                pass
