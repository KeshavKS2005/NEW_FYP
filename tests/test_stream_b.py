"""
Unit tests for Stream B: Known-Object Detection (Module 2).
Verifies contract compliance, coordinate constraints, confidence gating,
and single-object isolation.
"""

import unittest
from pathlib import Path
import sys
import numpy as np
import cv2

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from stream_b.config import StreamBConfig, KnownBoundingBox, StreamBFrameResult
from stream_b.detector import YOLOKnownDetector


class TestStreamB(unittest.TestCase):
    """Test suite for Stream B Known-Object Detector."""

    @classmethod
    def setUpClass(cls):
        cls.weights_path = workspace_root / "stream_b" / "weights" / "known_single_object.pt"
        if not cls.weights_path.exists():
            raise unittest.SkipTest(f"Weights file not found at {cls.weights_path}")

        cls.config = StreamBConfig(
            model_path=str(cls.weights_path),
            conf_threshold=0.40,
            device="0",
            known_classes=["bicycle"]
        )
        cls.detector = YOLOKnownDetector(config=cls.config)

    def test_empty_frame_handling(self):
        """Verifies handling of blank/empty frames without crashing."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(blank_frame, frame_index=1)

        self.assertIsInstance(result, StreamBFrameResult)
        self.assertEqual(result.frame_index, 1)
        self.assertIsInstance(result.boxes, list)
        self.assertIsInstance(result.box_matrix, np.ndarray)
        self.assertEqual(result.box_matrix.shape, (0, 4))
        self.assertEqual(result.known_count, 0)

    def test_output_contract_on_bicycle_image(self):
        """Verifies output matrix shape [K, 4] and coordinate validity on real bicycle sample."""
        val_img_dir = workspace_root / "datasets" / "single_object_dataset" / "images" / "val"
        val_images = list(val_img_dir.glob("*.jpg"))
        if not val_images:
            self.skipTest("No validation images found")

        # Find an image that yields detections
        detected = False
        for img_path in val_images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            result = self.detector.detect(frame, frame_index=42)

            if result.known_count > 0:
                detected = True
                # Check contract types
                self.assertIsInstance(result.box_matrix, np.ndarray)
                self.assertEqual(result.box_matrix.ndim, 2)
                self.assertEqual(result.box_matrix.shape[1], 4)
                self.assertEqual(len(result.boxes), result.box_matrix.shape[0])

                # Check box coordinate bounds and sanity
                for box in result.boxes:
                    self.assertGreaterEqual(box.x1, 0)
                    self.assertGreaterEqual(box.y1, 0)
                    self.assertLessEqual(box.x2, w)
                    self.assertLessEqual(box.y2, h)
                    self.assertLess(box.x1, box.x2)
                    self.assertLess(box.y1, box.y2)

                    # Check confidence gating
                    self.assertGreaterEqual(box.confidence, self.config.conf_threshold)

                    # Check single class restriction
                    self.assertEqual(box.class_id, 0)
                    self.assertEqual(box.class_name, "bicycle")
                break

        self.assertTrue(detected, "At least one validation image should contain bicycle detections")

    def test_zero_detections_on_cctv_frame(self):
        """
        Verifies that on CCTV surveillance video (where only pedestrians and cars exist),
        Stream B produces ZERO detections, preserving pedestrians and cars for Module 3 erasure.
        """
        cctv_video = workspace_root / "CCTV 01" / "VIRAT_S_010001_00_000019_000065.mp4"
        if not cctv_video.exists():
            self.skipTest(f"CCTV video not found at {cctv_video}")

        cap = cv2.VideoCapture(str(cctv_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ret, frame = cap.read()
        cap.release()

        self.assertTrue(ret, "Failed to read CCTV video frame")
        result = self.detector.detect(frame, frame_index=50)

        # Confirm 0 known detections (pedestrians and cars are NOT known!)
        self.assertEqual(result.known_count, 0)
        self.assertEqual(result.box_matrix.shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
