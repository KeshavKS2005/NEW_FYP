"""
Module 4: Temporal Trajectory Gater (Noise Filtering & Active Learning)
Adheres strictly to fyp_architecture.md Section 3 Module 4 & Section 5.
Filters transient noise and stationary oscillations, verifies physical persistence,
and triggers the Active Learning edge-case mining hook.
"""

import math
import os
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union, Any
import cv2
import numpy as np

from module_4.config import Module4Config, ConfirmedUnknownTrack, Module4FrameResult
from module_4.byte_tracker import ByteTracker, STrack


class TemporalTrajectoryGater:
    """
    Temporal Trajectory Gating Engine:
    1. Associates candidate unknown boxes across frames via ByteTrack.
    2. Performs Directional Persistence Check to flag stationary oscillating noise (e.g., swaying foliage).
    3. Implements Frame Survival Gating (W = 5 consecutive hits required to confirm track).
    4. Triggers Active Learning Hook (auto-crops original frame when track survives > 30 frames).
    """

    def __init__(self, config: Optional[Module4Config] = None):
        self.config = config or Module4Config()
        self.tracker = ByteTracker(
            track_thresh=self.config.track_thresh,
            match_thresh=self.config.match_thresh,
            max_lost_frames=self.config.max_lost_frames
        )
        self._cropped_track_ids: Set[int] = set()

        if self.config.active_learning:
            Path(self.config.active_learning_crop_dir).mkdir(parents=True, exist_ok=True)

    def reset(self):
        """Resets tracker state and active learning cache."""
        self.tracker.reset()
        self._cropped_track_ids.clear()

    @staticmethod
    def _compute_trajectory_metrics(trajectory: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        Computes (net_displacement, total_path_length) from centroid history.
        """
        if len(trajectory) < 2:
            return 0.0, 0.0

        # Net displacement from start to latest
        start_pt = trajectory[0]
        curr_pt = trajectory[-1]
        net_disp = math.hypot(curr_pt[0] - start_pt[0], curr_pt[1] - start_pt[1])

        # Cumulative path length along trajectory
        total_path = 0.0
        for i in range(1, len(trajectory)):
            p1 = trajectory[i - 1]
            p2 = trajectory[i]
            total_path += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        return net_disp, total_path

    def update(
        self,
        candidate_boxes: Union[List[Any], np.ndarray],
        frame: Optional[np.ndarray] = None,
        frame_index: int = 0
    ) -> Module4FrameResult:
        """
        Updates temporal gating and tracking with candidate unknown boxes.

        Args:
            candidate_boxes: List of CandidateUnknownBox or [U, 5] numpy array of (x1, y1, x2, y2, conf).
            frame: Optional (H, W, 3) raw RGB/BGR frame for Active Learning cropping.
            frame_index: Sequential integer index of current frame.

        Returns:
            Module4FrameResult with confirmed tracks, all tracks, and saved crops.
        """
        t0 = time.perf_counter()

        # Format input into [N, 5] matrix
        if isinstance(candidate_boxes, np.ndarray):
            if candidate_boxes.size == 0:
                matrix = np.empty((0, 5), dtype=np.float32)
            elif candidate_boxes.ndim == 1 and len(candidate_boxes) >= 4:
                conf = candidate_boxes[4] if len(candidate_boxes) >= 5 else 1.0
                matrix = np.array([[candidate_boxes[0], candidate_boxes[1], candidate_boxes[2], candidate_boxes[3], conf]], dtype=np.float32)
            else:
                matrix = candidate_boxes.astype(np.float32)
        elif isinstance(candidate_boxes, (list, tuple)):
            rows = []
            for b in candidate_boxes:
                if hasattr(b, "as_tracker_detection"):
                    rows.append(b.as_tracker_detection())
                elif hasattr(b, "x1") and hasattr(b, "y1") and hasattr(b, "x2") and hasattr(b, "y2"):
                    conf = getattr(b, "confidence", 1.0)
                    rows.append([float(b.x1), float(b.y1), float(b.x2), float(b.y2), float(conf)])
                elif isinstance(b, (list, tuple)) and len(b) >= 4:
                    conf = float(b[4]) if len(b) >= 5 else 1.0
                    rows.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), conf])
            matrix = np.array(rows, dtype=np.float32) if rows else np.empty((0, 5), dtype=np.float32)
        else:
            matrix = np.empty((0, 5), dtype=np.float32)

        # 1. Update ByteTracker with candidate boxes
        active_stracks = self.tracker.update(matrix)

        all_tracks: List[ConfirmedUnknownTrack] = []
        confirmed_tracks: List[ConfirmedUnknownTrack] = []
        saved_crops: List[str] = []

        # 2. Evaluate Temporal Gating & Directional Persistence for each track
        for strack in active_stracks:
            tlbr = strack.tlbr
            x1, y1, x2, y2 = int(round(tlbr[0])), int(round(tlbr[1])), int(round(tlbr[2])), int(round(tlbr[3]))
            centroid = strack.centroid

            net_disp, path_len = self._compute_trajectory_metrics(strack.trajectory)

            # Directional Persistence Check (Noise Filtering)
            is_noise = False
            if self.config.directional_persistence and len(strack.trajectory) >= self.config.survival_window:
                if path_len >= 10.0:
                    oscillation_ratio = net_disp / max(1e-4, path_len)
                    # If high total movement but net displacement near zero, object is oscillating in place
                    if (oscillation_ratio < self.config.max_oscillation_ratio and
                            net_disp < self.config.min_net_displacement):
                        is_noise = True

            # Frame Survival Gating: requires W consecutive hits and non-noise
            is_confirmed = (
                strack.consecutive_hits >= self.config.survival_window and
                not is_noise
            )

            track_obj = ConfirmedUnknownTrack(
                track_id=strack.track_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=strack.score,
                age=strack.tracklet_len,
                hits=strack.hits,
                consecutive_hits=strack.consecutive_hits,
                centroid=centroid,
                trajectory=list(strack.trajectory),
                net_displacement=net_disp,
                total_path_length=path_len,
                is_confirmed=is_confirmed,
                is_noise=is_noise
            )

            all_tracks.append(track_obj)
            if is_confirmed:
                confirmed_tracks.append(track_obj)

                # 3. Active Learning Hook (fyp_architecture.md Section 5)
                # If an unknown track survives > 30 frames, auto-crop original RGB frame
                if (self.config.active_learning and
                        strack.hits >= self.config.active_learning_frame_thresh and
                        strack.track_id not in self._cropped_track_ids and
                        frame is not None and frame.size > 0):

                    fh, fw = frame.shape[:2]
                    pad = self.config.active_learning_crop_padding
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(fw, x2 + pad)
                    cy2 = min(fh, y2 + pad)

                    if cx2 > cx1 and cy2 > cy1:
                        crop_img = frame[cy1:cy2, cx1:cx2]
                        crop_filename = f"unknown_track_{strack.track_id:04d}_frame_{frame_index:05d}.jpg"
                        crop_path = os.path.join(self.config.active_learning_crop_dir, crop_filename)
                        cv2.imwrite(crop_path, crop_img)
                        saved_crops.append(crop_path)
                        self._cropped_track_ids.add(strack.track_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return Module4FrameResult(
            frame_index=frame_index,
            confirmed_tracks=confirmed_tracks,
            all_active_tracks=all_tracks,
            saved_crops=saved_crops,
            processing_time_ms=elapsed_ms
        )
