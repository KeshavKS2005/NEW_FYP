"""
End-to-End 12-Module Open-World Continual Discovery & Retraining Pipeline
Adheres strictly to fyp_architecture_final_object.md.

Multiprocessing Architecture:
1. Phase 1: Real-Time 30+ FPS Discrepancy Tracker & Live Reload Loop (Modules 1-4)
2. Phase 2: Asynchronous Compound Decomposition, Clustering & VLM Discovery Worker (Modules 5-9)
3. Phase 3: Continual YOLO Retraining, Evaluation Gate & Hot-Swap Worker (Modules 10-12)
"""

import os
import sys
import time
import shutil
from pathlib import Path
from multiprocessing import Process, Queue, Event
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from stream_a.motion_extractor import MOG2MotionExtractor
from stream_b.detector import YOLOKnownDetector
from stream_b.config import StreamBConfig
from module_3.erasure_engine import MaskErasureEngine
from module_3.config import Module3Config
from module_4.temporal_gater import TemporalTrajectoryGater
from module_4.config import Module4Config
from module_5.compound_decomposer import CompoundBlobDecomposer
from module_5.config import Module5Config
from module_6.dinov2_extractor import DINOv2FeatureExtractor
from module_6.config import Module6Config
from module_7.faiss_clustering import FAISSClusterEngine
from module_7.config import Module7Config
from module_8.db_manager import DatabaseManager
from module_8.config import Module8Config, load_phase1_env
from module_9.vlm_identifier import VLMDiscoveryIdentifier
from module_9.config import Module9Config
from module_10.dataset_builder import ContinualDatasetBuilder
from module_10.config import Module10Config
from module_11.trainer_evaluator import IncrementalTrainerAndGate
from module_11.config import Module11Config
from module_12.hot_swapper import AtomicModelRegistryAndHotSwapper
from module_12.config import Module12Config


# Load credentials from FYP_phase1/.env
load_phase1_env()

CHAMPION_WEIGHTS_PATH = os.environ.get("CHAMPION_WEIGHTS_PATH", "models/champion_latest.pt")
VIDEO_SOURCE = os.environ.get("SURVEILLANCE_STREAM_URL", "datasets/test_videos/bike_traffic_montreal.webm")
HOLDOUT_CLASS_IDS = [
    int(x) for x in os.environ.get("TEST_HOLDOUT_CLASS_IDS", "").split(",") if x.strip().isdigit()
]


def discovery_worker_loop(discovery_queue: Queue, retrain_queue: Queue):
    """
    Phase 2 Async Discovery Worker (Modules 5, 6, 7, 8, and 9).
    """
    print("[Phase 2 Worker] Starting downstream discovery worker...")
    db = DatabaseManager()
    faiss_engine = FAISSClusterEngine()
    decomposer = CompoundBlobDecomposer()
    embedder = DINOv2FeatureExtractor()
    vlm = VLMDiscoveryIdentifier()

    os.makedirs("storage/frames", exist_ok=True)
    os.makedirs("storage/crops", exist_ok=True)
    os.makedirs("storage/masked_crops", exist_ok=True)

    while True:
        item = discovery_queue.get()
        if item is None:
            print("[Phase 2 Worker] Shutting down.")
            break

        track_id, frame_bgr, raw_crop, mask_crop, merged_bbox, sharpness = item

        # MODULE 5: Compound Blob Decomposition
        sub_objects = decomposer.decompose(frame_bgr, raw_crop, mask_crop, merged_bbox)
        frame_hw = frame_bgr.shape[:2]

        for sub_idx, sub_obj in enumerate(sub_objects):
            bx1, by1, bx2, by2 = sub_obj.bbox
            s_raw = sub_obj.raw_crop
            s_mask = sub_obj.sub_mask
            method_used = sub_obj.method

            # MODULE 6: Neighbor & Background Suppression + DINOv2 Vector
            processed = embedder.process_sub_object(
                sub_raw=s_raw,
                sub_mask=s_mask,
                abs_bbox=(bx1, by1, bx2, by2),
                frame_hw=frame_hw,
                method_label=method_used
            )

            emb = processed["embedding"]
            sub_sharpness = processed["sharpness"]

            # Save clean pristine full frame for Module 10 (NEVER with red box drawn on it)
            frame_path = f"storage/frames/track_{track_id}_sub_{sub_idx}.jpg"
            crop_path = f"storage/crops/track_{track_id}_sub_{sub_idx}.jpg"
            masked_path = f"storage/masked_crops/track_{track_id}_sub_{sub_idx}.jpg"

            cv2.imwrite(frame_path, frame_bgr)
            cv2.imwrite(crop_path, s_raw)
            cv2.imwrite(masked_path, processed["masked_crop"])

            # MODULE 7: FAISS Centroid Matching & Dynamic Updates
            best_faiss_id, best_sim, is_new = faiss_engine.match_or_add(emb)

            # MODULE 8: PostgreSQL Observation Store
            if is_new:
                cluster_id, obs_count, status = db.insert_cluster(best_faiss_id)
            else:
                cluster_id, obs_count, status = db.increment_cluster(best_faiss_id)

            db.insert_observation(
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

            # MODULE 9: Automated Gemini 2.5 Flash VLM Trigger
            if vlm.should_trigger(obs_count, status):
                print(f"[Phase 2 Worker] Triggering Gemini 2.5 Flash on cluster {cluster_id} (count={obs_count})...")
                vlm.identify_cluster(cluster_id=cluster_id, db=db, retrain_queue=retrain_queue)

    db.close()


def continual_retraining_worker(retrain_queue: Queue, model_reload_event: Event):
    """
    Phase 3 Continual Retraining & Hot-Swap Worker (Modules 10, 11, and 12).
    """
    print("[Phase 3 Worker] Starting continual retraining worker...")
    db = DatabaseManager()
    dataset_builder = ContinualDatasetBuilder()
    trainer_gate = IncrementalTrainerAndGate()
    hot_swapper = AtomicModelRegistryAndHotSwapper()

    while True:
        cluster_id = retrain_queue.get()
        if cluster_id is None:
            print("[Phase 3 Worker] Shutting down.")
            break

        cluster_info = db.get_cluster(cluster_id)
        if not cluster_info or cluster_info.get("status") != "IDENTIFIED":
            continue

        print(f"[Phase 3 Worker] Processing identified cluster {cluster_id} ('{cluster_info.get('identified_label')}')...")
        db.update_cluster_status(cluster_id=cluster_id, status="RETRAINING")

        try:
            # MODULE 10: Build Incremental Dataset with Teacher Pseudo-Labeler
            ds_yaml_path, old_class_ids, target_cls_id, new_label = dataset_builder.build_dataset(
                cluster_id=cluster_id,
                champion_weights_path=CHAMPION_WEIGHTS_PATH,
                db=db
            )

            # MODULE 11: Fine-Tune Challenger & Evaluate Metric Gates
            passed_gates, challenger_weights, metrics = trainer_gate.evaluate_and_train(
                champion_weights_path=CHAMPION_WEIGHTS_PATH,
                dataset_yaml_path=ds_yaml_path,
                old_class_ids=old_class_ids,
                target_cls_id=target_cls_id,
                cluster_id=cluster_id
            )

            # MODULE 12: Atomic Model Registry & Zero-Downtime Hot-Swap
            hot_swapper.deploy_or_reject(
                challenger_weights_path=challenger_weights,
                promote=passed_gates,
                metrics=metrics,
                cluster_id=cluster_id,
                target_cls_id=target_cls_id,
                db=db,
                model_reload_event=model_reload_event
            )
        except Exception as e:
            print(f"[Phase 3 Worker] Error during continual cycle for cluster {cluster_id}: {e}")
            db.update_cluster_status(cluster_id=cluster_id, status="IDENTIFIED")

    db.close()


def run_surveillance_pipeline(video_path: Optional[str] = None, max_frames: Optional[int] = None):
    """
    Phase 1 Real-Time Discrepancy Tracking Loop (Modules 1, 2, 3, 4) with Atomic Hot-Swap Reload.
    """
    stream_url = video_path or VIDEO_SOURCE
    print(f"[Phase 1 Tracker] Launching video stream: {stream_url}")

    discovery_queue = Queue(maxsize=256)
    retrain_queue = Queue(maxsize=64)
    model_reload_event = Event()

    discovery_proc = Process(
        target=discovery_worker_loop, args=(discovery_queue, retrain_queue), daemon=True
    )
    retrain_proc = Process(
        target=continual_retraining_worker, args=(retrain_queue, model_reload_event), daemon=True
    )
    discovery_proc.start()
    retrain_proc.start()

    # Module 1: Stream A
    motion_extractor = MOG2MotionExtractor()

    # Module 2: Stream B with champion weights & holdout classes
    stream_b_config = StreamBConfig(
        model_path=CHAMPION_WEIGHTS_PATH,
        holdout_class_ids=set(HOLDOUT_CLASS_IDS)
    )
    detector = YOLOKnownDetector(stream_b_config)

    # Module 3: Mask Erasure Engine with pad_px = 15
    erasure_config = Module3Config(pad_px=15, min_contour_area=500.0)
    erasure_engine = MaskErasureEngine(erasure_config)

    # Module 4: Temporal Trajectory Gater
    gater_config = Module4Config(survival_window=8, min_dir_persistence=0.50, max_lost_frames=5)
    gater = TemporalTrajectoryGater(gater_config)

    cap = cv2.VideoCapture(stream_url)
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Live atomic reload check from Module 12
            if model_reload_event.is_set():
                print("[Phase 1 Tracker] Hot-swap signal received! Reloading Champion weights in-place...")
                detector.reload_weights(CHAMPION_WEIGHTS_PATH)
                model_reload_event.clear()

            # MODULE 1: Stream A Motion Extraction
            motion_res = motion_extractor.extract(frame, frame_idx)

            # MODULE 2: Stream B Known-Object Detection
            known_res = detector.detect(frame, frame_idx)

            # MODULE 3: Mask-Level Erasure
            erased_res = erasure_engine.process(motion_res.motion_mask, known_res.boxes, frame_idx)

            # MODULE 4: Trajectory Gating & Keyframe Buffering
            gater_res = gater.update(
                candidate_boxes=erased_res.boxes,
                frame=frame,
                frame_index=frame_idx,
                fg_mask=erased_res.erased_mask,
                work_queue=discovery_queue
            )

    finally:
        cap.release()
        print("[Phase 1 Tracker] Video stream finished. Draining worker queues...")
        discovery_queue.put(None)
        retrain_queue.put(None)
        discovery_proc.join(timeout=5)
        retrain_proc.join(timeout=5)
        print("[Phase 1 Tracker] All processes cleanly stopped.")


if __name__ == "__main__":
    run_surveillance_pipeline()
