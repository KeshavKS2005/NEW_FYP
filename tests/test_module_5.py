"""
Tests for Module 5: Compound Blob Decomposer & Keyframe Buffer
"""

import os
import shutil
import pytest
import numpy as np
import cv2
from PIL import Image

from module_5.config import Module5Config, DecomposedSubObject
from module_5.compound_decomposer import CompoundBlobDecomposer
from module_4.keyframe_buffer import compute_sharpness, TrackKeyframeBuffer


class TestModule5:
    """Test suite for Module 5 decomposition and keyframe sharpness buffering."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_dir = "tests/test_outputs_module5"
        self.config = Module5Config(
            min_sub_object_area=100,
            part_min_fg_ratio=0.04,
            part_merge_cosine_thresh=0.62,
            maskcut_sep_thresh=0.22,
            device="cpu"
        )
        yield
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sharpness_scoring(self):
        """Crisp edges must yield significantly higher sharpness variance than blurred edges."""
        sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp_img[::10, :] = 255
        sharp_img[:, ::10] = 255

        blurred_img = cv2.GaussianBlur(sharp_img, (21, 21), 5.0)

        sharp_score = compute_sharpness(sharp_img)
        blurred_score = compute_sharpness(blurred_img)

        assert sharp_score > blurred_score
        assert blurred_score >= 0.0

    def test_empty_crop_sharpness(self):
        """Empty or degenerate crops must safely return 0.0 without throwing exceptions."""
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        assert compute_sharpness(empty) == 0.0
        assert compute_sharpness(None) == 0.0

    def test_keyframe_buffer_tracks_best(self):
        """TrackKeyframeBuffer must retain the payload with maximum sharpness across frames."""
        buffer = TrackKeyframeBuffer()

        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        mask1 = np.ones((100, 100), dtype=np.uint8) * 255
        s1 = buffer.update(track_id=1, bbox=(10, 10, 50, 50), frame_bgr=frame1, fg_mask=mask1)

        frame2 = frame1.copy()
        frame2[10:50:4, 10:50] = 255
        s2 = buffer.update(track_id=1, bbox=(10, 10, 50, 50), frame_bgr=frame2, fg_mask=mask1)

        frame3 = frame1.copy()
        s3 = buffer.update(track_id=1, bbox=(10, 10, 50, 50), frame_bgr=frame3, fg_mask=mask1)

        best_payload = buffer.get_best_payload(track_id=1)
        assert best_payload is not None
        best_frame, best_raw, best_mask, best_bbox, best_sharp = best_payload

        assert best_sharp == s2
        assert s2 > s1

        popped = buffer.pop_best_payload(track_id=1)
        assert popped is not None
        assert buffer.get_best_payload(track_id=1) is None

    def test_compound_decomposer_single_blob(self):
        """Small or single-component blob safely returns 1 DecomposedSubObject."""
        decomposer = CompoundBlobDecomposer(self.config)

        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        raw_crop = np.full((60, 60, 3), 200, dtype=np.uint8)
        mask_crop = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(mask_crop, (30, 30), 15, 255, -1)

        sub_objects = decomposer.decompose(
            frame_bgr=frame,
            raw_crop=raw_crop,
            mask_crop=mask_crop,
            merged_bbox=(50, 50, 110, 110)
        )

        assert isinstance(sub_objects, list)
        assert len(sub_objects) >= 1
        assert isinstance(sub_objects[0], DecomposedSubObject)
        assert len(sub_objects[0].bbox) == 4
        assert sub_objects[0].sub_mask.shape == mask_crop.shape

    def test_compound_decomposer_contract(self):
        """DecomposedSubObject contract compliance."""
        sub = DecomposedSubObject(
            bbox=(10, 20, 60, 80),
            raw_crop=np.zeros((60, 50, 3), dtype=np.uint8),
            sub_mask=np.ones((60, 50), dtype=np.uint8) * 255,
            method="METHOD2_FASTSAM_DINOV2"
        )
        assert sub.bbox == (10, 20, 60, 80)
        assert sub.method == "METHOD2_FASTSAM_DINOV2"
        assert sub.raw_crop.shape == (60, 50, 3)
