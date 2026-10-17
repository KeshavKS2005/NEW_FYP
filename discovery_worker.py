"""
Phase 2: Downstream Asynchronous Discovery Worker Loop
Orchestrates Modules 5, 6, 7, 8, and 9 in an asynchronous background process/thread
to ensure zero FPS drop on the upstream real-time video pipeline.

Data Flow:
Queue -> Module 5 (FastSAM+DINOv2 Decomposer) -> Module 6 (Neighbor Masking & DINOv2)
      -> Module 7 (FAISS + Disk Sync) -> Module 8 (DB Store) -> Module 9 (Gemini 2.5 Flash)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List
from multiprocessing import Queue
import cv2
import numpy as np

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from module_5.config import Module5Config
from module_5.compound_decomposer import CompoundBlobDecomposer
from module_6.config import Module6Config
from module_6.dinov2_extractor import DINOv2FeatureExtractor
from module_7.config import Module7Config
from module_7.faiss_clustering import FAISSClusterEngine
from module_8.config import Module8Config
from module_8.db_manager import DatabaseManager
from module_9.config import Module9Config
from module_9.vlm_identifier import VLMDiscoveryIdentifier


class DiscoveryWorker:
    """
    Worker manager that executes the Phase 2 downstream discovery pipeline.
    """

    def __init__(
        self,
        m5_config: Optional[Module5Config] = None,
        m6_config: Optional[Module6Config] = None,
        m7_config: Optional[Module7Config] = None,
        m8_config: Optional[Module8Config] = None,
        m9_config: Optional[Module9Config] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        self.decomposer = CompoundBlobDecomposer(m5_config)
        self.embedder = DINOv2FeatureExtractor(m6_config)
        self.db = db_manager or DatabaseManager(m8_config)
        self.faiss_engine = FAISSClusterEngine(m7_config, db_manager=self.db)
        self.vlm = VLMDiscoveryIdentifier(m9_config)

        os.makedirs("storage/frames", exist_ok=True)
        os.makedirs("storage/crops", exist_ok=True)
        os.makedirs("storage/masked_crops", exist_ok=True)

    def process_item(
        self,
        item: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Any, float],
        return_list: bool = False,
        retrain_queue: Optional[Any] = None
    ) -> Any:
        """
        Processes a single confirmed completed track payload through Modules 5-9.

        Args:
            item: (track_id, frame_bgr, raw_crop, mask_crop, bbox, sharpness)

        Returns:
            List of dictionaries with discovery metadata for each decomposed sub-object.
        """
        track_id, frame_bgr, raw_crop, mask_crop, bbox, sharpness = item
        frame_hw = frame_bgr.shape[:2]

        # 1. Module 5: Decompose compound blob into isolated sub-objects
        sub_objects = self.decomposer.decompose(frame_bgr, raw_crop, mask_crop, bbox)
        results = []

        for sub_idx, sub_obj in enumerate(sub_objects):
            bx1, by1, bx2, by2 = sub_obj.bbox
            s_raw = sub_obj.raw_crop
            s_mask = sub_obj.sub_mask
            method_used = sub_obj.method

            # 2. Module 6: Mask out background & neighbors, extract 384-D vector
            processed = self.embedder.process_sub_object(
                sub_raw=s_raw,
                sub_mask=s_mask,
                abs_bbox=(bx1, by1, bx2, by2),
                frame_hw=frame_hw,
                method_label=method_used
            )

            emb = processed["embedding"]
            sub_sharpness = processed["sharpness"]

            # Save clean pristine full frame (NEVER with red box drawn on it)
            frame_path = f"storage/frames/track_{track_id}_sub_{sub_idx}.jpg"
            crop_path = f"storage/crops/track_{track_id}_sub_{sub_idx}.jpg"
            masked_path = f"storage/masked_crops/track_{track_id}_sub_{sub_idx}.jpg"

            cv2.imwrite(frame_path, frame_bgr)
            cv2.imwrite(crop_path, s_raw)
            cv2.imwrite(masked_path, processed["masked_crop"])

            # 3. Module 7: FAISS Centroid Matching & Dynamic Updates
            best_faiss_id, best_sim, is_new = self.faiss_engine.match_or_add(emb)

            # 4. Module 8: Database logging
            if is_new:
                cluster_id, obs_count, status = self.db.insert_cluster(best_faiss_id)
            else:
                cluster_id, obs_count, status = self.db.increment_cluster(best_faiss_id)

            self.db.insert_observation(
                cluster_id=cluster_id,
                track_id=track_id,
                crop_image_path=crop_path,
                full_frame_path=frame_path,
                bbox_xyxy=[bx1, by1, bx2, by2],
                sharpness_score=sub_sharpness,
                cosine_similarity=best_sim,
                sub_object_idx=sub_idx,
                masked_crop_path=masked_path,
                frame_hw=list(frame_hw),
                decomposition_method=method_used
            )

            # 5. Module 9: Check VLM Discovery trigger condition
            vlm_result = None
            if self.vlm.should_trigger(obs_count, status):
                vlm_result = self.vlm.identify_cluster(
                    cluster_id=cluster_id, db=self.db, retrain_queue=retrain_queue
                )

            results.append({
                "track_id": track_id,
                "cluster_id": cluster_id,
                "observation_count": obs_count,
                "status": status,
                "similarity": best_sim,
                "vlm_result": vlm_result,
                "frame_path": frame_path,
                "crop_path": crop_path
            })

        if return_list:
            return results
        return results[0] if len(results) == 1 else results


def discovery_worker_loop(
    queue: Queue,
    m5_config: Optional[Module5Config] = None,
    m6_config: Optional[Module6Config] = None,
    m7_config: Optional[Module7Config] = None,
    m8_config: Optional[Module8Config] = None,
    m9_config: Optional[Module9Config] = None,
    result_queue: Optional[Queue] = None,
    retrain_queue: Optional[Queue] = None
):
    """
    Daemon worker process function that continuously drains the work queue.
    Terminates cleanly when receiving None.
    """
    worker = DiscoveryWorker(
        m5_config=m5_config,
        m6_config=m6_config,
        m7_config=m7_config,
        m8_config=m8_config,
        m9_config=m9_config
    )

    while True:
        try:
            item = queue.get()
            if item is None:
                break

            results = worker.process_item(item, return_list=True, retrain_queue=retrain_queue)

            if result_queue is not None:
                for res in results:
                    result_queue.put(res)

        except Exception as e:
            print(f"[DiscoveryWorker] Error processing item: {e}")

    worker.db.close()
