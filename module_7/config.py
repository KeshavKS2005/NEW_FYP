"""
Module 7 Configuration: Online FAISS Centroid Clustering Engine
"""

from dataclasses import dataclass
import os


@dataclass
class Module7Config:
    """Configuration parameters for FAISS clustering engine."""

    # DINOv2 feature vector dimensionality
    embedding_dim: int = 384

    # Cosine similarity threshold to merge with nearest centroid
    cosine_threshold: float = 0.78

    # Paths for disk index persistence & startup synchronization
    index_path: str = "storage/faiss_centroids.index"
    centroids_path: str = "storage/centroids.npy"

    # PostgreSQL DSN & SQLite fallback (for database manager integration)
    postgres_dsn: str = os.environ.get(
        "POSTGRES_DSN",
        "dbname=surveillance user=postgres password=secret host=localhost"
    )
    use_sqlite_fallback: bool = True
    sqlite_db_path: str = "outputs/surveillance_discovery.db"
