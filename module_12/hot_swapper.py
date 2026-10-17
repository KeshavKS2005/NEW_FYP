"""
Module 12: Atomic Model Registry & Zero-Downtime Hot-Swapper
Atomically replaces production weights using os.replace(), records evaluation metrics
in the model_registry database table, updates cluster status, and signals Phase 1 to hot-swap.
Adheres strictly to Module 12 in fyp_architecture_final_object.md.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from multiprocessing import Event
from ultralytics import YOLO

from module_12.config import Module12Config
from module_8.db_manager import DatabaseManager


class AtomicModelRegistryAndHotSwapper:
    """
    Manages production weight promotion, database audit logging, and live reload signaling.
    """

    def __init__(self, config: Optional[Module12Config] = None):
        self.config = config or Module12Config()

    def deploy_or_reject(
        self,
        challenger_weights_path: Path,
        promote: bool,
        metrics: Dict[str, Any],
        cluster_id: int,
        target_cls_id: int,
        db: DatabaseManager,
        model_reload_event: Optional[Event] = None
    ) -> int:
        """
        Logs evaluation record into model_registry and atomically swaps weights if approved.

        Args:
            challenger_weights_path: Path to best.pt produced by Module 11.
            promote: Decision boolean from Module 11 evaluation gates.
            metrics: Metrics dictionary containing old and new class mAP values.
            cluster_id: ID of the cluster.
            target_cls_id: Sequential YOLO class ID assigned to the new class.
            db: Relational database manager.
            model_reload_event: Optional multiprocessing.Event signaling Phase 1 live reload.

        Returns:
            model_version_id recorded in database.
        """
        # Load challenger class names
        try:
            chal_model = YOLO(str(challenger_weights_path))
            class_names = dict(chal_model.names) if hasattr(chal_model, "names") and chal_model.names else {}
        except Exception:
            class_names = {}

        # 1. Log evaluation in model_registry table
        model_version_id = db.insert_model_registry(
            weights_path=str(challenger_weights_path),
            num_classes=len(class_names),
            class_names=class_names,
            old_classes_map50_95=metrics.get("chal_old_map", 0.0),
            new_class_map50=metrics.get("new_cls_map50", 0.0),
            new_class_recall=metrics.get("new_cls_recall", 0.0),
            promoted_to_production=promote,
            trained_on_cluster_id=cluster_id
        )

        if promote:
            print(f"[Module 12] Gates PASSED! Promoting challenger {challenger_weights_path} to production...")
            champ_path = Path(self.config.champion_weights_path)
            staging_path = Path(self.config.staging_weights_path)
            champ_path.parent.mkdir(parents=True, exist_ok=True)

            # Staging copy followed by atomic file replacement
            shutil.copy(str(challenger_weights_path), staging_path)
            os.replace(staging_path, champ_path)

            # Update cluster status in DB
            db.update_cluster_status(
                cluster_id=cluster_id,
                status="DEPLOYED_IN_YOLO",
                assigned_yolo_class_id=target_cls_id
            )

            # Signal Phase 1 (30 FPS loop) to reload weights immediately
            if model_reload_event is not None:
                model_reload_event.set()
                print("[Module 12] Live hot-swap signal (model_reload_event) dispatched to Phase 1.")
        else:
            print(f"[Module 12] Gates REJECTED for cluster {cluster_id}. Retaining existing Champion weights.")
            db.update_cluster_status(
                cluster_id=cluster_id,
                status="IDENTIFIED"
            )

        return model_version_id


ModelHotSwapper = AtomicModelRegistryAndHotSwapper
