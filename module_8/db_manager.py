"""
Module 8: Relational Database Persistence Manager
Supports PostgreSQL (psycopg2) as defined in schema.sql, with automatic SQLite fallback
for development and test environments where local PostgreSQL credentials differ.
Adheres strictly to fyp_architecture_final_object.md Section 7 & 8.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from module_8.config import Module8Config


class DatabaseManager:
    """
    Unified relational persistence manager for unknown clusters, observations, and model registry.
    """

    def __init__(self, config: Optional[Module8Config] = None):
        self.config = config or Module8Config()
        self.backend = "postgres"
        self._conn = None
        self._connect()
        self._init_schema()

    def _connect(self):
        """Attempts to connect to PostgreSQL; falls back to SQLite on connection failure or if custom sqlite_db_path / force_sqlite is set."""
        is_custom_sqlite = self.config.sqlite_db_path != "outputs/surveillance_discovery.db"
        if getattr(self.config, "force_sqlite", False) or os.environ.get("USE_SQLITE", "0") == "1" or is_custom_sqlite:
            Path(self.config.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.config.sqlite_db_path, check_same_thread=False)
            self.backend = "sqlite"
            return

        try:
            import psycopg2
            self._conn = psycopg2.connect(self.config.postgres_dsn)
            self._conn.autocommit = False
            self.backend = "postgres"
        except Exception as e:
            if not self.config.use_sqlite_fallback:
                raise RuntimeError(f"PostgreSQL connection failed and fallback disabled: {e}")

            Path(self.config.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.config.sqlite_db_path, check_same_thread=False)
            self.backend = "sqlite"

    def _init_schema(self):
        """Creates the unknown_clusters, unknown_observations, and model_registry tables."""
        cur = self._conn.cursor()
        if self.backend == "postgres":
            cur.execute("""
                CREATE TABLE IF NOT EXISTS unknown_clusters (
                    cluster_id SERIAL PRIMARY KEY,
                    faiss_index_id INT UNIQUE NOT NULL,
                    observation_count INT NOT NULL DEFAULT 1,
                    status VARCHAR(32) NOT NULL DEFAULT 'COLLECTING',
                    identified_label VARCHAR(128) DEFAULT NULL,
                    assigned_yolo_class_id INT DEFAULT NULL,
                    vlm_confidence FLOAT DEFAULT NULL,
                    vlm_reasoning TEXT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS unknown_observations (
                    observation_id SERIAL PRIMARY KEY,
                    cluster_id INT NOT NULL REFERENCES unknown_clusters(cluster_id) ON DELETE CASCADE,
                    track_id INT NOT NULL,
                    sub_object_idx INT NOT NULL DEFAULT 0,
                    crop_image_path TEXT NOT NULL,
                    masked_crop_path TEXT NOT NULL,
                    full_frame_path TEXT NOT NULL,
                    bbox_xyxy INT[4] NOT NULL,
                    frame_hw INT[2] NOT NULL,
                    sharpness_score FLOAT NOT NULL,
                    cosine_similarity FLOAT NOT NULL,
                    decomposition_method VARCHAR(32) NOT NULL,
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    model_version_id SERIAL PRIMARY KEY,
                    weights_path TEXT NOT NULL,
                    num_classes INT NOT NULL,
                    class_names_json TEXT NOT NULL,
                    old_classes_map50_95 FLOAT NOT NULL,
                    new_class_map50 FLOAT NOT NULL,
                    new_class_recall FLOAT NOT NULL,
                    promoted_to_production BOOLEAN NOT NULL DEFAULT FALSE,
                    trained_on_cluster_id INT REFERENCES unknown_clusters(cluster_id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_cluster_sharpness 
                ON unknown_observations(cluster_id, sharpness_score DESC);
            """)
        else:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS unknown_clusters (
                    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    faiss_index_id INTEGER UNIQUE NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'COLLECTING',
                    identified_label TEXT DEFAULT NULL,
                    assigned_yolo_class_id INTEGER DEFAULT NULL,
                    vlm_confidence REAL DEFAULT NULL,
                    vlm_reasoning TEXT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS unknown_observations (
                    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL REFERENCES unknown_clusters(cluster_id) ON DELETE CASCADE,
                    track_id INTEGER NOT NULL,
                    sub_object_idx INTEGER NOT NULL DEFAULT 0,
                    crop_image_path TEXT NOT NULL,
                    masked_crop_path TEXT NOT NULL,
                    full_frame_path TEXT NOT NULL,
                    bbox_xyxy TEXT NOT NULL,
                    frame_hw TEXT NOT NULL,
                    sharpness_score REAL NOT NULL,
                    cosine_similarity REAL NOT NULL,
                    decomposition_method TEXT NOT NULL,
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    model_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weights_path TEXT NOT NULL,
                    num_classes INTEGER NOT NULL,
                    class_names_json TEXT NOT NULL,
                    old_classes_map50_95 REAL NOT NULL,
                    new_class_map50 REAL NOT NULL,
                    new_class_recall REAL NOT NULL,
                    promoted_to_production INTEGER NOT NULL DEFAULT 0,
                    trained_on_cluster_id INTEGER REFERENCES unknown_clusters(cluster_id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_cluster_sharpness 
                ON unknown_observations(cluster_id, sharpness_score DESC);
            """)
        self._conn.commit()

    def insert_cluster(self, faiss_index_id: int) -> Tuple[int, int, str]:
        """Inserts a new cluster record initialized with observation_count = 1."""
        cur = self._conn.cursor()
        if self.backend == "postgres":
            cur.execute("""
                INSERT INTO unknown_clusters (faiss_index_id, observation_count)
                VALUES (%s, 1)
                RETURNING cluster_id, observation_count, status;
            """, (faiss_index_id,))
            row = cur.fetchone()
        else:
            cur.execute("""
                INSERT INTO unknown_clusters (faiss_index_id, observation_count)
                VALUES (?, 1);
            """, (faiss_index_id,))
            cluster_id = cur.lastrowid
            cur.execute("""
                SELECT cluster_id, observation_count, status
                FROM unknown_clusters WHERE cluster_id = ?;
            """, (cluster_id,))
            row = cur.fetchone()

        self._conn.commit()
        return int(row[0]), int(row[1]), str(row[2])

    def increment_cluster(self, faiss_index_id: int) -> Tuple[int, int, str]:
        """Increments observation_count and updates timestamp for existing cluster."""
        cur = self._conn.cursor()
        if self.backend == "postgres":
            cur.execute("""
                UPDATE unknown_clusters
                SET observation_count = observation_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE faiss_index_id = %s
                RETURNING cluster_id, observation_count, status;
            """, (faiss_index_id,))
            row = cur.fetchone()
        else:
            cur.execute("""
                UPDATE unknown_clusters
                SET observation_count = observation_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE faiss_index_id = ?;
            """, (faiss_index_id,))
            cur.execute("""
                SELECT cluster_id, observation_count, status
                FROM unknown_clusters WHERE faiss_index_id = ?;
            """, (faiss_index_id,))
            row = cur.fetchone()

        self._conn.commit()
        return int(row[0]), int(row[1]), str(row[2])

    def update_cluster_status(
        self,
        cluster_id: int,
        status: str,
        identified_label: Optional[str] = None,
        assigned_yolo_class_id: Optional[int] = None,
        vlm_confidence: Optional[float] = None,
        vlm_reasoning: Optional[str] = None
    ) -> None:
        """Updates status, label, YOLO class ID, and reasoning for a cluster."""
        cur = self._conn.cursor()
        if self.backend == "postgres":
            cur.execute("""
                UPDATE unknown_clusters
                SET status = %s,
                    identified_label = COALESCE(%s, identified_label),
                    assigned_yolo_class_id = COALESCE(%s, assigned_yolo_class_id),
                    vlm_confidence = COALESCE(%s, vlm_confidence),
                    vlm_reasoning = COALESCE(%s, vlm_reasoning),
                    updated_at = CURRENT_TIMESTAMP
                WHERE cluster_id = %s;
            """, (status, identified_label, assigned_yolo_class_id, vlm_confidence, vlm_reasoning, cluster_id))
        else:
            cur.execute("""
                UPDATE unknown_clusters
                SET status = ?,
                    identified_label = COALESCE(?, identified_label),
                    assigned_yolo_class_id = COALESCE(?, assigned_yolo_class_id),
                    vlm_confidence = COALESCE(?, vlm_confidence),
                    vlm_reasoning = COALESCE(?, vlm_reasoning),
                    updated_at = CURRENT_TIMESTAMP
                WHERE cluster_id = ?;
            """, (status, identified_label, assigned_yolo_class_id, vlm_confidence, vlm_reasoning, cluster_id))
        self._conn.commit()

    def insert_observation(
        self,
        cluster_id: int,
        track_id: int,
        crop_image_path: str,
        full_frame_path: str,
        bbox_xyxy: List[int],
        sharpness_score: float,
        cosine_similarity: float,
        sub_object_idx: int = 0,
        masked_crop_path: Optional[str] = None,
        frame_hw: Optional[List[int]] = None,
        decomposition_method: str = "SINGLE"
    ) -> int:
        """Logs an observation record linked to a cluster."""
        cur = self._conn.cursor()
        masked_path = masked_crop_path or crop_image_path
        hw = frame_hw or [1080, 1920]

        if self.backend == "postgres":
            cur.execute("""
                INSERT INTO unknown_observations
                (cluster_id, track_id, sub_object_idx, crop_image_path, masked_crop_path,
                 full_frame_path, bbox_xyxy, frame_hw, sharpness_score,
                 cosine_similarity, decomposition_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING observation_id;
            """, (
                cluster_id, track_id, sub_object_idx, crop_image_path, masked_path,
                full_frame_path, bbox_xyxy, hw, sharpness_score,
                cosine_similarity, decomposition_method
            ))
            row = cur.fetchone()
            obs_id = int(row[0])
        else:
            cur.execute("""
                INSERT INTO unknown_observations
                (cluster_id, track_id, sub_object_idx, crop_image_path, masked_crop_path,
                 full_frame_path, bbox_xyxy, frame_hw, sharpness_score,
                 cosine_similarity, decomposition_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                cluster_id, track_id, sub_object_idx, crop_image_path, masked_path,
                full_frame_path, json.dumps(bbox_xyxy), json.dumps(hw),
                sharpness_score, cosine_similarity, decomposition_method
            ))
            obs_id = cur.lastrowid

        self._conn.commit()
        return obs_id

    def get_top_observations(self, cluster_id: int, limit: int = 3) -> List[Tuple[str, str, List[int]]]:
        """Returns list of (crop_image_path, full_frame_path, bbox_xyxy) ordered by sharpness."""
        cur = self._conn.cursor()
        if self.backend == "postgres":
            cur.execute("""
                SELECT crop_image_path, full_frame_path, bbox_xyxy
                FROM unknown_observations
                WHERE cluster_id = %s
                ORDER BY sharpness_score DESC
                LIMIT %s;
            """, (cluster_id, limit))
            rows = cur.fetchall()
            return [(r[0], r[1], list(r[2])) for r in rows]
        else:
            cur.execute("""
                SELECT crop_image_path, full_frame_path, bbox_xyxy
                FROM unknown_observations
                WHERE cluster_id = ?
                ORDER BY sharpness_score DESC
                LIMIT ?;
            """, (cluster_id, limit))
            rows = cur.fetchall()
            results = []
            for r in rows:
                b = json.loads(r[2]) if isinstance(r[2], str) else list(r[2])
                results.append((r[0], r[1], b))
            return results

    def get_cluster_observations(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Returns all observation records for dataset building."""
        cur = self._conn.cursor()
        if self.backend == "postgres":
            cur.execute("""
                SELECT observation_id, full_frame_path, bbox_xyxy, frame_hw
                FROM unknown_observations
                WHERE cluster_id = %s
                ORDER BY sharpness_score DESC;
            """, (cluster_id,))
            rows = cur.fetchall()
            return [
                {
                    "observation_id": r[0],
                    "full_frame_path": r[1],
                    "bbox_xyxy": list(r[2]),
                    "frame_hw": list(r[3])
                }
                for r in rows
            ]
        else:
            cur.execute("""
                SELECT observation_id, full_frame_path, bbox_xyxy, frame_hw
                FROM unknown_observations
                WHERE cluster_id = ?
                ORDER BY sharpness_score DESC;
            """, (cluster_id,))
            rows = cur.fetchall()
            results = []
            for r in rows:
                bbox = json.loads(r[2]) if isinstance(r[2], str) else list(r[2])
                hw = json.loads(r[3]) if isinstance(r[3], str) else list(r[3])
                results.append({
                    "observation_id": r[0],
                    "full_frame_path": r[1],
                    "bbox_xyxy": bbox,
                    "frame_hw": hw
                })
            return results

    def insert_model_registry(
        self,
        weights_path: str,
        num_classes: int,
        class_names: Dict[int, str],
        old_classes_map50_95: float,
        new_class_map50: float,
        new_class_recall: float,
        promoted_to_production: bool,
        trained_on_cluster_id: int
    ) -> int:
        """Logs evaluation metrics into model_registry."""
        cur = self._conn.cursor()
        names_json = json.dumps(class_names)
        cluster_fk = trained_on_cluster_id if (trained_on_cluster_id and trained_on_cluster_id > 0) else None
        if self.backend == "postgres":
            cur.execute("""
                INSERT INTO model_registry
                (weights_path, num_classes, class_names_json, old_classes_map50_95,
                 new_class_map50, new_class_recall, promoted_to_production, trained_on_cluster_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING model_version_id;
            """, (
                weights_path, num_classes, names_json, old_classes_map50_95,
                new_class_map50, new_class_recall, promoted_to_production, cluster_fk
            ))
            row = cur.fetchone()
            m_id = int(row[0])
        else:
            cur.execute("""
                INSERT INTO model_registry
                (weights_path, num_classes, class_names_json, old_classes_map50_95,
                 new_class_map50, new_class_recall, promoted_to_production, trained_on_cluster_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                weights_path, num_classes, names_json, old_classes_map50_95,
                new_class_map50, new_class_recall, int(promoted_to_production), cluster_fk
            ))
            m_id = cur.lastrowid

        self._conn.commit()
        return m_id

    def insert_model_version(
        self,
        weights_path: str,
        num_classes: int,
        class_names: Any,
        old_classes_map50_95: float,
        new_class_map50: float,
        new_class_recall: float,
        promoted_to_production: bool = False,
        trained_on_cluster_id: Optional[int] = None
    ) -> int:
        """Alias for insert_model_registry."""
        if isinstance(class_names, list):
            class_names_dict = {i: name for i, name in enumerate(class_names)}
        else:
            class_names_dict = class_names
        return self.insert_model_registry(
            weights_path=weights_path,
            num_classes=num_classes,
            class_names=class_names_dict,
            old_classes_map50_95=old_classes_map50_95,
            new_class_map50=new_class_map50,
            new_class_recall=new_class_recall,
            promoted_to_production=promoted_to_production,
            trained_on_cluster_id=trained_on_cluster_id
        )

    def promote_model_version(self, model_version_id: int) -> None:
        """Promotes a model version to production and demotes others."""
        cur = self._conn.cursor()
        if self.backend == "postgres":
            cur.execute("UPDATE model_registry SET promoted_to_production = FALSE;")
            cur.execute("UPDATE model_registry SET promoted_to_production = TRUE WHERE model_version_id = %s;", (model_version_id,))
        else:
            cur.execute("UPDATE model_registry SET promoted_to_production = FALSE;")
            cur.execute("UPDATE model_registry SET promoted_to_production = TRUE WHERE model_version_id = ?;", (model_version_id,))
        self._conn.commit()

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """Fetches the currently promoted production model."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT model_version_id, weights_path, num_classes, class_names_json,
                   old_classes_map50_95, new_class_map50, new_class_recall, created_at
            FROM model_registry
            WHERE promoted_to_production = TRUE
            ORDER BY model_version_id DESC
            LIMIT 1;
        """)
        row = cur.fetchone()
        if not row:
            return None
        return {
            "model_version_id": row[0],
            "weights_path": row[1],
            "num_classes": row[2],
            "class_names": json.loads(row[3]) if isinstance(row[3], str) else row[3],
            "old_classes_map50_95": row[4],
            "new_class_map50": row[5],
            "new_class_recall": row[6],
            "created_at": str(row[7])
        }

    def get_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a cluster by ID."""
        cur = self._conn.cursor()
        q = """
            SELECT cluster_id, faiss_index_id, observation_count, status,
                   identified_label, assigned_yolo_class_id, vlm_confidence, vlm_reasoning
            FROM unknown_clusters WHERE cluster_id = 
        """
        param = "%s" if self.backend == "postgres" else "?"
        cur.execute(q + param + ";", (cluster_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "cluster_id": row[0],
            "faiss_index_id": row[1],
            "observation_count": row[2],
            "status": row[3],
            "identified_label": row[4],
            "assigned_yolo_class_id": row[5],
            "vlm_confidence": row[6],
            "vlm_reasoning": row[7]
        }

    def close(self):
        """Closes the underlying database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
