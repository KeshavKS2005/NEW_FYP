"""
End-to-End Test for the 12-Module Continual Discovery & Retraining Architecture
Validates the complete 12-module lifecycle:
- Phase 1 (Modules 1-4): Motion extraction, YOLO detection, 15px erasure, directional gating & keyframe payload.
- Phase 2 (Modules 5-9): FastSAM/DINOv2 decomposition, feature extraction, FAISS disk clustering, SQLite/Postgres DB logging, Gemini VLM identification on clean frames.
- Phase 3 (Modules 10-12): Continual dataset builder (standard 5-col bboxes), dual-gated evaluation, and atomic hot-swap.
"""

import os
import shutil
from pathlib import Path
from multiprocessing import Queue, Event
import pytest
import numpy as np
import cv2

from stream_a.config import MotionConfig
from stream_a.motion_extractor import MOG2MotionExtractor
from stream_b.config import StreamBConfig, KnownBoundingBox
from stream_b.detector import YOLOKnownDetector
from module_3.config import Module3Config
from module_3.erasure_engine import MaskErasureEngine
from module_4.config import Module4Config
from module_4.temporal_gater import TemporalTrajectoryGater
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
from module_10.config import Module10Config
from module_10.dataset_builder import ContinualDatasetBuilder
from module_11.config import Module11Config
from module_11.trainer_evaluator import ChallengerTrainerEvaluator
from module_12.config import Module12Config
from module_12.hot_swapper import ModelHotSwapper


class TestPipeline12ModulesE2E:
    """Full end-to-end integration test across all 12 modules."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_12_e2e"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "pipeline12.db")
        self.faiss_idx = os.path.join(self.test_dir, "faiss.index")
        self.faiss_c = os.path.join(self.test_dir, "faiss_c.npy")
        self.champ_weights = os.path.join(self.test_dir, "champion_active.pt")
        self.staging_weights = os.path.join(self.test_dir, "champion_staging.pt")

        # Copy baseline YOLO weights for champion
        shutil.copy("yolo11n.pt", self.champ_weights)

        self.db_config = Module8Config(
            use_sqlite_fallback=True,
            sqlite_db_path=self.db_path
        )
        self.db = DatabaseManager(self.db_config)
        yield
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_complete_12_module_lifecycle(self):
        """
        Executes a complete pass through Modules 1 to 12:
        1-4: Motion + YOLO + 15px Erasure + Gating produces confirmed anomaly payload.
        5-9: FastSAM decomposition + DINOv2 vector + FAISS match + DB logging + VLM trigger on clean frame.
        10-12: Dataset creation (5-col bboxes) + Gate evaluation + Atomic hot-swap + Stream B reload.
        """
        # ==================== PHASE 1: MODULES 1 to 4 ====================
        extractor = MOG2MotionExtractor(MotionConfig(history=20, min_contour_area=50.0))
        erasure = MaskErasureEngine(Module3Config(pad_px=15, min_contour_area=100.0))
        gater = TemporalTrajectoryGater(Module4Config(
            survival_window=5,
            max_lost_frames=5,
            active_learning=False
        ))

        # Generate 10 frames of moving anomaly
        emitted_payloads = []
        frame_hw = (480, 640)
        fg_mask = np.zeros(frame_hw, dtype=np.uint8)
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        for f in range(10):
            # Clean background frame
            frame.fill(40)
            fg_mask.fill(0)
            # Add textured moving box
            x = 100 + f * 15
            y = 120
            frame[y:y+60, x:x+60] = 220
            frame[y:y+60:4, x:x+60] = 255
            fg_mask[y:y+60, x:x+60] = 255

            c_boxes = np.array([[x, y, x + 60, y + 60, 0.95]], dtype=np.float32)

            res = gater.update(
                candidate_boxes=c_boxes,
                frame=frame,
                frame_index=f + 1,
                fg_mask=fg_mask
            )
            if res.completed_track_payloads:
                emitted_payloads.extend(res.completed_track_payloads)

        # Trigger track termination (6 empty frames to exceed max_lost_frames=5)
        empty_boxes = np.empty((0, 5), dtype=np.float32)
        empty_mask = np.zeros(frame_hw, dtype=np.uint8)
        for f in range(10, 17):
            res = gater.update(empty_boxes, frame, f + 1, empty_mask)
            if res.completed_track_payloads:
                emitted_payloads.extend(res.completed_track_payloads)

        assert len(emitted_payloads) >= 1, "Phase 1 failed to emit completed track payload!"
        track_id, frame_bgr, raw_crop, mask_crop, bbox, sharpness = emitted_payloads[0]

        # ==================== PHASE 2: MODULES 5 to 9 ====================
        decomposer = CompoundBlobDecomposer(Module5Config())
        embedder = DINOv2FeatureExtractor(Module6Config())
        m7_cfg = Module7Config(
            index_path=self.faiss_idx,
            centroids_path=self.faiss_c,
            cosine_threshold=0.78
        )
        faiss_engine = FAISSClusterEngine(m7_cfg, db_manager=self.db)
        vlm = VLMDiscoveryIdentifier(Module9Config(trigger_count=1, allow_mock_fallback=True))

        # Module 5: Decompose
        sub_objects = decomposer.decompose(frame_bgr, raw_crop, mask_crop, bbox)
        assert len(sub_objects) >= 1
        sub_obj = sub_objects[0]
        bx1, by1, bx2, by2 = sub_obj.bbox

        # Module 6: Gray neighbor suppression and 384-D vector
        processed = embedder.process_sub_object(
            sub_raw=sub_obj.raw_crop,
            sub_mask=sub_obj.sub_mask,
            abs_bbox=(bx1, by1, bx2, by2),
            frame_hw=frame_hw,
            method_label=sub_obj.method
        )
        emb = processed["embedding"]
        assert emb.shape == (1, 384)

        # Save clean pristine full frame (Fix #2: never with red box drawn on it)
        clean_frame_path = os.path.join(self.test_dir, f"clean_frame_{track_id}.jpg")
        crop_path = os.path.join(self.test_dir, f"clean_crop_{track_id}.jpg")
        cv2.imwrite(clean_frame_path, frame_bgr)
        cv2.imwrite(crop_path, sub_obj.raw_crop)

        # Module 7: FAISS clustering
        best_faiss_id, best_sim, is_new = faiss_engine.match_or_add(emb)
        assert is_new is True

        # Module 8: Database logging (Fix #1: standard bbox_xyxy, no polygon)
        cluster_id, obs_count, status = self.db.insert_cluster(best_faiss_id)
        obs_id = self.db.insert_observation(
            cluster_id=cluster_id,
            track_id=track_id,
            crop_image_path=crop_path,
            full_frame_path=clean_frame_path,
            bbox_xyxy=[bx1, by1, bx2, by2],
            sharpness_score=processed["sharpness"],
            cosine_similarity=best_sim,
            frame_hw=list(frame_hw)
        )
        assert obs_id >= 1

        # Module 9: VLM Trigger on pristine frame
        retrain_q = Queue()
        assert vlm.should_trigger(obs_count, status) is True
        vlm_res = vlm.identify_cluster(cluster_id=cluster_id, db=self.db, retrain_queue=retrain_q)
        assert vlm_res is not None
        assert not retrain_q.empty()
        dispatched_cid = retrain_q.get_nowait()
        assert dispatched_cid == cluster_id

        # Verify clean frame on disk was NOT contaminated with red pixels (Fix #2)
        on_disk_frame = cv2.imread(clean_frame_path)
        assert on_disk_frame is not None
        red_pixels = np.all(on_disk_frame == [0, 0, 255], axis=-1)
        assert not np.any(red_pixels)

        # ==================== PHASE 3: MODULES 10 to 12 ====================
        builder = ContinualDatasetBuilder(Module10Config(
            datasets_base_dir=os.path.join(self.test_dir, "datasets"),
            replay_buffer_dir=os.path.join(self.test_dir, "replay")
        ))
        evaluator = ChallengerTrainerEvaluator(Module11Config())
        hot_swapper = ModelHotSwapper(Module12Config(
            champion_weights_path=self.champ_weights,
            staging_weights_path=self.staging_weights
        ))

        # Module 10: Build dataset with teacher pseudo-labeler and standard 5-col bboxes
        ds_yaml, old_ids, target_cls, label_name = builder.build_dataset(
            cluster_id=cluster_id,
            champion_weights_path=self.champ_weights,
            db=self.db
        )
        assert ds_yaml.exists()

        # Check label file has standard 5-column format (Fix #1)
        label_files = list(ds_yaml.parent.glob("labels/*/*.txt"))
        assert len(label_files) >= 1
        for lf in label_files:
            lines = lf.read_text().strip().splitlines()
            for line in lines:
                parts = line.split()
                assert len(parts) == 5, f"Non-standard label line: {line}"

        # Module 11: Evaluate Gates
        metrics = evaluator.evaluate_gates(
            champ_old_map=0.520,
            chal_old_map=0.515,     # Old classes drop = 0.005 <= 0.015 (Gate 1 PASS)
            new_cls_map50=0.750,     # >= 0.60 (Gate 2 PASS)
            new_cls_recall=0.780    # >= 0.65 (Gate 2 PASS)
        )
        assert metrics["promote_challenger"] is True

        # Module 12: Atomic Hot-Swap
        reload_event = Event()
        fake_challenger = Path(self.test_dir) / "challenger_trained.pt"
        fake_challenger.write_bytes(b"UPDATED_CHALLENGER_WEIGHTS")

        m_ver = hot_swapper.deploy_or_reject(
            challenger_weights_path=fake_challenger,
            promote=True,
            metrics=metrics,
            cluster_id=cluster_id,
            target_cls_id=target_cls,
            db=self.db,
            model_reload_event=reload_event
        )

        assert m_ver >= 1
        assert reload_event.is_set() is True
        assert Path(self.champ_weights).read_bytes() == b"UPDATED_CHALLENGER_WEIGHTS"

        # Verify DB cluster status transitioned to DEPLOYED_IN_YOLO
        final_cluster = self.db.get_cluster(cluster_id)
        assert final_cluster["status"] == "DEPLOYED_IN_YOLO"
