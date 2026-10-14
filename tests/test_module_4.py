"""
Unit tests for Module 4: Temporal Trajectory Gating (Noise Filtering & ByteTrack).
Verifies:
- Kalman filter prediction and state correction.
- ByteTrack association and track ID consistency across frames.
- Frame Survival Gating (W = 5 consecutive frames).
- Directional Persistence Check (rejects stationary oscillating noise).
- 1-frame glitch elimination.
- Active Learning edge-case mining hook (> 30 frames auto-crop).
"""

import shutil
import unittest
from pathlib import Path
import sys
import numpy as np
import cv2

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from module_4.config import Module4Config, ConfirmedUnknownTrack, Module4FrameResult
from module_4.kalman_filter import KalmanBoxFilter
from module_4.byte_tracker import ByteTracker, STrack
from module_4.temporal_gater import TemporalTrajectoryGater


class TestModule4(unittest.TestCase):
    """Test suite for Module 4 Temporal Trajectory Gating and ByteTrack."""

    def setUp(self):
        self.crop_test_dir = workspace_root / "outputs" / "test_active_learning_crops"
        if self.crop_test_dir.exists():
            shutil.rmtree(self.crop_test_dir)

        self.config = Module4Config(
            survival_window=5,
            max_lost_frames=10,
            track_thresh=0.5,
            match_thresh=0.7,
            directional_persistence=True,
            min_net_displacement=15.0,
            max_oscillation_ratio=0.30,
            active_learning=True,
            active_learning_frame_thresh=30,
            active_learning_crop_dir=str(self.crop_test_dir)
        )
        self.gater = TemporalTrajectoryGater(config=self.config)

    def tearDown(self):
        if self.crop_test_dir.exists():
            shutil.rmtree(self.crop_test_dir)

    def test_kalman_box_filter(self):
        """Verifies Kalman filter initialization, projection, and update steps."""
        kf = KalmanBoxFilter()
        measurement = np.array([100.0, 150.0, 1.5, 40.0], dtype=np.float32)

        mean, cov = kf.initiate(measurement)
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(cov.shape, (8, 8))

        pred_mean, pred_cov = kf.predict(mean, cov)
        self.assertEqual(pred_mean.shape, (8,))

        # Update with new measurement slightly moved
        new_meas = np.array([102.0, 151.0, 1.5, 40.0], dtype=np.float32)
        up_mean, up_cov = kf.update(pred_mean, pred_cov, new_meas)
        self.assertAlmostEqual(up_mean[0], 102.0, delta=1.0)

    def test_frame_survival_gating(self):
        """
        Verifies that a tracklet MUST survive for W=5 consecutive frames
        before being confirmed as an Unknown Moving Object.
        """
        # Feed detection moving smoothly for 6 frames:
        # Frames 1..4: unconfirmed (0 confirmed tracks)
        # Frame 5+: confirmed (1 confirmed track with persistent ID)
        initial_track_id = None

        for f in range(1, 7):
            x1 = 100 + f * 5
            y1 = 150 + f * 5
            x2 = x1 + 50
            y2 = y1 + 50
            cand_box = [x1, y1, x2, y2, 1.0]

            res = self.gater.update([cand_box], frame_index=f)

            if f < 5:
                # Still in warm-up survival window
                self.assertEqual(res.confirmed_count, 0, f"Frame {f} should NOT be confirmed yet")
                self.assertEqual(len(res.all_active_tracks), 1)
            else:
                # Survived W=5 frames!
                self.assertEqual(res.confirmed_count, 1, f"Frame {f} MUST be confirmed")
                confirmed = res.confirmed_tracks[0]
                self.assertTrue(confirmed.is_confirmed)
                if initial_track_id is None:
                    initial_track_id = confirmed.track_id
                else:
                    # Verify ID persistence
                    self.assertEqual(confirmed.track_id, initial_track_id)

    def test_transient_glitch_elimination(self):
        """
        Verifies that a 1-frame transient noise detection (e.g. camera auto-exposure flash)
        is never output as a confirmed unknown object.
        """
        # Frame 1: Glitch appears
        glitch = [50, 50, 90, 90, 1.0]
        res1 = self.gater.update([glitch], frame_index=1)
        self.assertEqual(res1.confirmed_count, 0)

        # Frames 2..10: No detections (blank)
        for f in range(2, 11):
            res = self.gater.update([], frame_index=f)
            self.assertEqual(res.confirmed_count, 0)

    def test_directional_persistence_rejects_oscillating_noise(self):
        """
        Verifies that a stationary oscillating object (like a swaying branch
        moving back and forth between 100 and 106 px) is rejected as noise.
        """
        # Alternate position back and forth:
        # Even frame: 100, Odd frame: 105
        # Total path distance will be large (> 30px), but net displacement will be <= 5px.
        for f in range(1, 15):
            offset = 5 if (f % 2 == 1) else 0
            box = [100 + offset, 200, 150 + offset, 250, 1.0]
            res = self.gater.update([box], frame_index=f)

            # Once the track has enough history to evaluate oscillation, it must be flagged as noise
            if f >= 8:
                # Should NOT be confirmed due to oscillation filter
                self.assertEqual(res.confirmed_count, 0, f"Frame {f} oscillating noise should be rejected")
                self.assertTrue(res.all_active_tracks[0].is_noise)

    def test_active_learning_crop_hook(self):
        """
        Verifies that when a confirmed unknown object survives for >= 30 frames,
        Module 4 automatically crops the original RGB bounding box and saves it to disk.
        """
        dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Draw a synthetic patch so the crop has distinct content
        cv2.rectangle(dummy_frame, (100, 100), (200, 200), (0, 255, 0), -1)

        saved_crops = []
        for f in range(1, 35):
            # Object moves with persistent velocity
            x1 = 100 + f * 2
            y1 = 100 + f * 1
            x2 = x1 + 50
            y2 = y1 + 50
            box = [x1, y1, x2, y2, 1.0]

            res = self.gater.update([box], frame=dummy_frame, frame_index=f)
            if res.saved_crops:
                saved_crops.extend(res.saved_crops)

        # Verify active learning triggered
        self.assertGreaterEqual(len(saved_crops), 1, "Active learning should have saved at least 1 crop")
        crop_path = Path(saved_crops[0])
        self.assertTrue(crop_path.exists(), f"Crop file does not exist: {crop_path}")
        self.assertGreater(crop_path.stat().st_size, 0, "Crop file is empty")

        # Read back saved image to verify validity
        crop_img = cv2.imread(str(crop_path))
        self.assertIsNotNone(crop_img)
        self.assertGreater(crop_img.shape[0], 0)
        self.assertGreater(crop_img.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
