"""
Unit tests for Module 3: Mask-Level Erasure & Extraction (The Discrepancy Engine).
Verifies:
- Mask-level erasure of known objects (Stream B).
- 10% dilation padding margin around known boxes.
- Survival of unknown moving objects as candidate boxes.
- Area filtering threshold (< 100 px discarded).
- Output contract compliance: [U, 5] numpy matrix formatted as (x1, y1, x2, y2, confidence=1.0).
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

from module_3.config import Module3Config, CandidateUnknownBox, Module3FrameResult
from module_3.erasure_engine import MaskErasureEngine
from stream_b.config import KnownBoundingBox


class TestModule3(unittest.TestCase):
    """Test suite for Module 3 Mask Erasure Engine."""

    def setUp(self):
        self.config = Module3Config(
            dilation_margin=0.10,
            min_contour_area=100.0,
            tracker_confidence=1.0,
            min_pad_px=2
        )
        self.engine = MaskErasureEngine(config=self.config)

    def test_empty_mask_handling(self):
        """Verifies graceful handling of empty or blank masks."""
        blank_mask = np.zeros((480, 640), dtype=np.uint8)
        result = self.engine.process(blank_mask, known_boxes=[], frame_index=1)

        self.assertIsInstance(result, Module3FrameResult)
        self.assertEqual(result.frame_index, 1)
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.box_matrix.shape, (0, 5))
        self.assertEqual(result.erased_known_count, 0)

    def test_known_object_erasure(self):
        """
        Verifies that when a known object's bounding box coincides with a motion blob,
        the motion blob is completely erased (0 active pixels remain).
        """
        h, w = 400, 600
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create a motion blob representing a moving bicycle at [100, 100, 160, 180]
        mask[100:180, 100:160] = 255

        known_box = KnownBoundingBox(
            x1=100, y1=100, x2=160, y2=180,
            confidence=0.95, class_id=0, class_name="bicycle"
        )

        result = self.engine.process(mask, known_boxes=[known_box], frame_index=10)

        # Known object should be erased -> 0 candidate unknown boxes
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(np.count_nonzero(result.erased_mask), 0)
        self.assertEqual(result.erased_known_count, 1)
        self.assertEqual(result.box_matrix.shape, (0, 5))

    def test_dilation_margin_covers_edge_motion(self):
        """
        Verifies that 10% dilation padding erases pixels slightly beyond the original YOLO box
        (e.g., protruding pedals or handlebars).
        """
        h, w = 400, 600
        mask = np.zeros((h, w), dtype=np.uint8)

        # Known box: width 100 (x: 200..300), height 100 (y: 200..300)
        # 10% dilation adds 10 px on every side -> covers x: 190..310, y: 190..310
        # Place small motion spots just outside the original box at x=195, y=250 (inside padded zone)
        mask[245:255, 192:198] = 255  # Inside padded margin
        mask[200:300, 200:300] = 255  # Inside known box

        known_box = [200, 200, 300, 300]

        result = self.engine.process(mask, known_boxes=[known_box], frame_index=1)

        # Everything inside padded box should be erased
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(np.count_nonzero(result.erased_mask), 0)

    def test_discrepancy_isolation(self):
        """
        Core thesis test:
        Frame has two moving objects:
        1. Known bicycle at (100, 100, 180, 180) -> ERASED
        2. Unknown anomaly (e.g., shopping cart / dog) at (350, 250, 420, 320) -> PRESERVED
        Module 3 must output exactly 1 Candidate Unknown Box at the anomaly coordinates.
        """
        h, w = 480, 640
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Bicycle motion (area = 80 x 80 = 6400 px)
        mask[100:180, 100:180] = 255

        # 2. Unknown anomaly motion (area = 70 x 70 = 4900 px)
        mask[250:320, 350:420] = 255

        # Stream B only detects the bicycle
        known_boxes = [KnownBoundingBox(
            x1=100, y1=100, x2=180, y2=180,
            confidence=0.89, class_id=0, class_name="bicycle"
        )]

        result = self.engine.process(mask, known_boxes=known_boxes, frame_index=42)

        # Verify only 1 candidate box survived
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.box_matrix.shape, (1, 5))

        cand = result.boxes[0]
        # Candidate box should match anomaly coordinates
        self.assertAlmostEqual(cand.x1, 350, delta=5)
        self.assertAlmostEqual(cand.y1, 250, delta=5)
        self.assertAlmostEqual(cand.x2, 420, delta=5)
        self.assertAlmostEqual(cand.y2, 320, delta=5)
        self.assertEqual(cand.confidence, 1.0)

        # Check [U, 5] matrix output values
        mat_row = result.box_matrix[0]
        self.assertEqual(mat_row[4], 1.0)
        self.assertEqual(int(mat_row[0]), cand.x1)
        self.assertEqual(int(mat_row[1]), cand.y1)

    def test_area_filtering_threshold(self):
        """
        Verifies that contours with area < min_contour_area (100 px) are discarded,
        while contours with area >= min_contour_area are retained.
        """
        h, w = 300, 300
        mask = np.zeros((h, w), dtype=np.uint8)

        # Tiny noise blob: 8x8 = 64 px (< 100 px)
        mask[20:28, 20:28] = 255

        # Valid small object: 12x12 = 144 px (>= 100 px)
        mask[100:112, 100:112] = 255

        result = self.engine.process(mask, known_boxes=[], frame_index=1)

        # Only the 144 px blob should survive
        self.assertEqual(result.candidate_count, 1)
        self.assertGreaterEqual(result.boxes[0].area, 100.0)

    def test_boundary_clipping(self):
        """Verifies that bounding boxes on the edges of the frame do not cause crashes or negative indices."""
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:20, 0:20] = 255

        # Known box at top-left corner
        known_boxes = [[0, 0, 25, 25]]
        result = self.engine.process(mask, known_boxes=known_boxes, frame_index=1)

        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(np.count_nonzero(result.erased_mask), 0)

    def test_fragmented_object_unification(self):
        """
        Verifies that fragmented motion blobs belonging to the same vehicle/object
        (e.g., front bumper at (100, 100, 150, 140) and rear at (165, 100, 215, 140) separated by 15px)
        are cleanly unified into a single candidate bounding box.
        """
        h, w = 400, 400
        mask = np.zeros((h, w), dtype=np.uint8)

        # Blob 1: front bumper (50 x 40 = 2000 px)
        mask[100:140, 100:150] = 255

        # Blob 2: rear bumper (50 x 40 = 2000 px), separated by 15 px gap along X (x: 165 to 215)
        mask[100:140, 165:215] = 255

        result = self.engine.process(mask, known_boxes=[], frame_index=1)

        # Must be merged into 1 single unified candidate box instead of 2!
        self.assertEqual(result.candidate_count, 1)
        unified = result.boxes[0]
        self.assertEqual(unified.x1, 100)
        self.assertEqual(unified.x2, 215)
        self.assertEqual(unified.y1, 100)
        self.assertEqual(unified.y2, 140)


if __name__ == "__main__":
    unittest.main()
