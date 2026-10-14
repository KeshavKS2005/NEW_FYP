"""
Unit tests for Stream A: Class-Agnostic Motion Extraction.
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

from stream_a.config import MotionConfig, MotionBoundingBox, MotionFrameResult
from stream_a.motion_extractor import MOG2MotionExtractor


class TestStreamA(unittest.TestCase):
    """Test suite for Stream A motion extraction module."""

    def setUp(self):
        self.config = MotionConfig(
            history=20,
            var_threshold=16.0,
            detect_shadows=True,
            morph_kernel_size=(5, 5),
            morph_open_iterations=1,
            morph_close_iterations=1,
            min_contour_area=100.0
        )
        self.extractor = MOG2MotionExtractor(config=self.config)

    def test_output_contract(self):
        """Verifies output dimensions, dtypes, and value constraints."""
        h, w = 360, 640
        frame = np.full((h, w, 3), 100, dtype=np.uint8)

        result = self.extractor.process(frame)

        self.assertIsInstance(result, MotionFrameResult)
        self.assertEqual(result.frame_idx, 1)
        self.assertEqual(result.raw_mask.shape, (h, w))
        self.assertEqual(result.clean_mask.shape, (h, w))
        self.assertEqual(result.clean_mask.dtype, np.uint8)

        # clean_mask must be strictly binary: only {0, 255}
        unique_vals = set(np.unique(result.clean_mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_shadow_elimination(self):
        """Verifies that MOG2 shadow pixels (value 127) are completely discarded."""
        # Directly test thresholding step:
        # A raw mask with background (0), shadow (127), and foreground (255)
        raw_mask = np.zeros((100, 100), dtype=np.uint8)
        raw_mask[20:40, 20:40] = 127   # Simulated shadow blob
        raw_mask[60:80, 60:80] = 255   # Simulated foreground object

        _, clean_mask = cv2.threshold(
            raw_mask,
            self.config.shadow_threshold - 1,
            255,
            cv2.THRESH_BINARY
        )

        # Shadow area should be zero
        self.assertEqual(np.count_nonzero(clean_mask[20:40, 20:40]), 0)
        # Foreground area should be preserved (20x20 = 400 pixels)
        self.assertEqual(np.count_nonzero(clean_mask[60:80, 60:80]), 400)

    def test_morphological_noise_suppression(self):
        """Verifies that isolated 1-pixel or 2-pixel salt noise is removed by 5x5 opening."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Inject random 1-pixel noise spikes
        mask[10, 10] = 255
        mask[30, 45] = 255
        mask[80, 75] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # All isolated spikes should be completely eliminated
        self.assertEqual(np.count_nonzero(opened), 0)

    def test_synthetic_moving_object_detection(self):
        """
        Feeds static background frames to let MOG2 stabilize,
        then introduces a moving 60x60 square to verify detection.
        """
        h, w = 480, 640
        # 1. Warm up the background model with static dark-gray frames
        for _ in range(15):
            static_frame = np.full((h, w, 3), 50, dtype=np.uint8)
            self.extractor.process(static_frame)

        # 2. Introduce moving bright rectangle at (200, 150) of size 60x60
        active_frame = np.full((h, w, 3), 50, dtype=np.uint8)
        active_frame[150:210, 200:260] = 230  # High contrast moving patch

        result = self.extractor.process(active_frame)

        # Motion should be detected
        self.assertGreater(result.total_motion_pixels, 0)
        self.assertGreater(len(result.boxes), 0)

        # Check detected bounding box location
        matched = False
        for box in result.boxes:
            # Overlap check with injected region [200, 150, 260, 210]
            if (box.x1 <= 210 and box.x2 >= 200 and
                box.y1 <= 160 and box.y2 >= 150):
                matched = True
                self.assertGreater(box.area, 500)
                break

        self.assertTrue(matched, "Moving synthetic object was not detected in bounding boxes.")

    def test_motion_bounding_box_methods(self):
        """Tests helper methods on MotionBoundingBox."""
        box = MotionBoundingBox(
            x1=10, y1=20, x2=50, y2=80,
            area=2400.0,
            centroid=(30, 50)
        )
        self.assertEqual(box.width, 40)
        self.assertEqual(box.height, 60)
        self.assertEqual(box.to_xyxy(), (10, 20, 50, 80))
        self.assertEqual(box.to_xywh(), (10, 20, 40, 60))


if __name__ == "__main__":
    unittest.main()
