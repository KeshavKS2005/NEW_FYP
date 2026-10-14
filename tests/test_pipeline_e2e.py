"""
End-to-End Pipeline Integration Tests.
Validates the complete 4-module pipeline:
Module 1: Stream A (MOG2 Motion Extraction)
Module 2: Stream B (YOLO Known-Object Detection)
Module 3: Mask-Level Erasure & Extraction (Discrepancy Engine)
Module 4: Temporal Trajectory Gating & Tracking (ByteTrack)
"""

import unittest
from pathlib import Path
import sys
import numpy as np
import cv2

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from stream_a.config import MotionConfig
from stream_a.motion_extractor import MOG2MotionExtractor
from stream_b.config import StreamBConfig, KnownBoundingBox
from stream_b.detector import YOLOKnownDetector
from module_3.config import Module3Config
from module_3.erasure_engine import MaskErasureEngine
from module_4.config import Module4Config
from module_4.temporal_gater import TemporalTrajectoryGater


class TestPipelineE2E(unittest.TestCase):
    """End-to-end integration test for the full 4-module discrepancy pipeline."""

    def setUp(self):
        # Module 1
        self.stream_a = MOG2MotionExtractor(MotionConfig(
            history=30, var_threshold=16.0, detect_shadows=True, min_contour_area=50.0
        ))
        # Module 3
        self.module_3 = MaskErasureEngine(Module3Config(
            dilation_margin=0.10, min_contour_area=100.0, min_pad_px=2
        ))
        # Module 4
        self.module_4 = TemporalTrajectoryGater(Module4Config(
            survival_window=5, active_learning=False
        ))

    def test_synthetic_dual_stream_discrepancy(self):
        """
        Synthesizes a 15-frame video sequence where:
        1. A known bicycle moves from (100, 100) to (170, 100).
        2. An unknown anomaly moves from (300, 300) to (370, 300).
        Verifies:
        - Stream A detects all motion.
        - Stream B emits known box for bicycle.
        - Module 3 erases bicycle from motion mask.
        - Module 4 ONLY tracks and confirms the unknown anomaly.
        """
        h, w = 480, 640

        # Warm up background model with 10 static frames
        for _ in range(10):
            bg = np.full((h, w, 3), 40, dtype=np.uint8)
            self.stream_a.process(bg)

        confirmed_unknown_ids = set()

        for step in range(1, 15):
            frame = np.full((h, w, 3), 40, dtype=np.uint8)

            # Known Bicycle: moves horizontally at y=100
            bx1 = 100 + step * 5
            by1 = 100
            bx2 = bx1 + 40
            by2 = by1 + 40
            frame[by1:by2, bx1:bx2] = 220  # bright moving patch

            # Unknown Anomaly: moves horizontally at y=300
            ax1 = 300 + step * 5
            ay1 = 300
            ax2 = ax1 + 40
            ay2 = ay1 + 40
            frame[ay1:ay2, ax1:ax2] = 220  # bright moving patch

            # 1. Stream A Motion Extraction
            res_a = self.stream_a.process(frame)

            # 2. Simulated Stream B (Detects strictly the bicycle)
            known_boxes = [KnownBoundingBox(
                x1=bx1, y1=by1, x2=bx2, y2=by2,
                confidence=0.92, class_id=0, class_name="bicycle"
            )]

            # 3. Module 3 Mask Erasure & Candidate Extraction
            res_3 = self.module_3.process(
                motion_mask=res_a.clean_mask,
                known_boxes=known_boxes,
                frame_index=step
            )

            # 4. Module 4 Temporal Trajectory Gating
            res_4 = self.module_4.update(
                candidate_boxes=res_3.box_matrix,
                frame=frame,
                frame_index=step
            )

            for trk in res_4.confirmed_tracks:
                confirmed_unknown_ids.add(trk.track_id)
                # Confirm bounding box is in the anomaly region (y around 300), NOT bicycle (y around 100)
                self.assertGreater(trk.y1, 200, "Bicycle was erroneously confirmed as an unknown object!")

        # Verify that an unknown object was successfully confirmed
        self.assertGreaterEqual(len(confirmed_unknown_ids), 1)

    def test_real_video_pipeline_run(self):
        """
        Runs the 4-module pipeline on real downloaded footage (bike_traffic_montreal.webm).
        Verifies:
        - Stream B detects real bicycles.
        - Module 3 executes erasure without exceptions.
        - Module 4 updates tracks and computes results.
        """
        video_path = workspace_root / "datasets" / "test_videos" / "bike_traffic_montreal.webm"
        if not video_path.exists():
            self.skipTest(f"Test video not found at {video_path}")

        weights_path = workspace_root / "stream_b" / "weights" / "known_single_object.pt"
        if not weights_path.exists():
            self.skipTest(f"Weights not found at {weights_path}")

        detector_b = YOLOKnownDetector(StreamBConfig(
            model_path=str(weights_path), conf_threshold=0.35, device="cpu"
        ))

        cap = cv2.VideoCapture(str(video_path))
        self.assertTrue(cap.isOpened(), "Could not open test video")

        total_known_erased = 0
        total_candidates = 0

        # Process first 25 frames
        for f_idx in range(25):
            ret, frame = cap.read()
            if not ret:
                break

            res_a = self.stream_a.process(frame)
            res_b = detector_b.detect(frame, frame_index=f_idx)
            res_3 = self.module_3.process(res_a.clean_mask, res_b.boxes, frame_index=f_idx)
            res_4 = self.module_4.update(res_3.box_matrix, frame=frame, frame_index=f_idx)

            total_known_erased += res_3.erased_known_count
            total_candidates += res_3.candidate_count

            # Integrity checks
            self.assertEqual(res_3.erased_mask.shape, frame.shape[:2])
            self.assertIsInstance(res_4.confirmed_tracks, list)

        cap.release()

        # In bike_traffic_montreal, bicycles are present and erased
        self.assertGreater(total_known_erased, 0, "Bicycles should be detected and erased in Montreal footage")


if __name__ == "__main__":
    unittest.main()
