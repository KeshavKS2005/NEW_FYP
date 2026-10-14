"""
Module 4: Standalone ByteTrack Tracker
Implements ByteTrack association logic with 2-stage matching and Kalman filtering.
Adheres to Module 4 specifications in fyp_architecture.md.
Pure Python + NumPy + SciPy implementation: zero external C-extension dependencies.
"""

from enum import IntEnum
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

from module_4.kalman_filter import KalmanBoxFilter


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Represents a single target tracklet maintained by the ByteTracker."""

    _count = 0

    def __init__(self, tlbr: np.ndarray, score: float = 1.0):
        # tlbr is [x1, y1, x2, y2]
        self._tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = float(score)

        self.kalman_filter = KalmanBoxFilter()
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.is_activated = False
        self.track_id = 0
        self.state = TrackState.New

        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.hits = 0
        self.consecutive_hits = 0
        self.time_since_update = 0

        # Trajectory history of (cx, cy)
        cx = int((self._tlbr[0] + self._tlbr[2]) / 2.0)
        cy = int((self._tlbr[1] + self._tlbr[3]) / 2.0)
        self.trajectory: List[Tuple[int, int]] = [(cx, cy)]

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_id(cls) -> None:
        cls._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        """Returns [top-left x, top-left y, width, height]."""
        tlbr = self.tlbr
        ret = tlbr.copy()
        ret[2] = ret[2] - ret[0]
        ret[3] = ret[3] - ret[1]
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Returns current bounding box [x1, y1, x2, y2]."""
        if self.mean is None:
            return self._tlbr.copy()
        cx, cy, a, h = self.mean[:4]
        w = max(1.0, a * h)
        h = max(1.0, h)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @property
    def centroid(self) -> Tuple[int, int]:
        b = self.tlbr
        return (int((b[0] + b[2]) / 2.0), int((b[1] + b[3]) / 2.0))

    @staticmethod
    def tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
        """Converts [x1, y1, x2, y2] to [cx, cy, aspect_ratio, height]."""
        w = max(1e-4, tlbr[2] - tlbr[0])
        h = max(1e-4, tlbr[3] - tlbr[1])
        cx = tlbr[0] + w / 2.0
        cy = tlbr[1] + h / 2.0
        a = w / h
        return np.array([cx, cy, a, h], dtype=np.float32)

    def activate(self, frame_id: int):
        """Activates a new tracklet with Kalman filter initialization."""
        self.track_id = self.next_id()
        measurement = self.tlbr_to_xyah(self._tlbr)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.hits = 1
        self.consecutive_hits = 1
        self.time_since_update = 0

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False):
        """Re-activates a lost tracklet with fresh observation."""
        measurement = self.tlbr_to_xyah(new_track.tlbr)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.hits += 1
        self.consecutive_hits += 1
        self.time_since_update = 0
        self.trajectory.append(self.centroid)

    def predict(self):
        """Projects track state forward using Kalman motion model."""
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = self.kalman_filter.predict(
                self.mean, self.covariance
            )
        self.time_since_update += 1

    def update(self, new_track: "STrack", frame_id: int):
        """Updates tracklet state using matched observation."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.hits += 1
        self.consecutive_hits += 1
        self.time_since_update = 0

        measurement = self.tlbr_to_xyah(new_track.tlbr)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.trajectory.append(self.centroid)

    def mark_lost(self):
        self.state = TrackState.Lost
        self.consecutive_hits = 0

    def mark_removed(self):
        self.state = TrackState.Removed


def compute_iou_batch(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two lists of bounding boxes [N, 4] and [M, 4].
    Returns (N, M) matrix of IoU values in [0.0, 1.0].
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)

    # atlbrs: [N, 4], btlbrs: [M, 4]
    ax1, ay1, ax2, ay2 = atlbrs[:, 0], atlbrs[:, 1], atlbrs[:, 2], atlbrs[:, 3]
    bx1, by1, bx2, by2 = btlbrs[:, 0], btlbrs[:, 1], btlbrs[:, 2], btlbrs[:, 3]

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    # Pairwise intersections: [N, M]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    union_area = np.maximum(1e-6, union_area)

    return inter_area / union_area


class ByteTracker:
    """
    Standalone ByteTrack tracker.
    Associates high-confidence detections first, then low-confidence detections.
    """

    def __init__(
        self,
        track_thresh: float = 0.50,
        match_thresh: float = 0.70,
        max_lost_frames: int = 30
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

    def reset(self):
        """Resets tracker state."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_id()

    def update(self, output_results: np.ndarray) -> List[STrack]:
        """
        Updates the tracker with detections from current frame.

        Args:
            output_results: [N, 5] matrix of (x1, y1, x2, y2, score).

        Returns:
            List of active confirmed/tracked STrack objects.
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        if output_results is not None and len(output_results) > 0:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]

            remain_inds = scores >= self.track_thresh
            inds_low = scores < self.track_thresh

            # High confidence detections
            detections = [
                STrack(bboxes[i], scores[i]) for i in range(len(output_results)) if remain_inds[i]
            ]
            # Low confidence detections
            detections_second = [
                STrack(bboxes[i], scores[i]) for i in range(len(output_results)) if inds_low[i]
            ]
        else:
            detections = []
            detections_second = []

        # Predict current locations with Kalman filter
        strack_pool = self.tracked_stracks + self.lost_stracks
        for strack in strack_pool:
            strack.predict()

        # Step 1: Match high-confidence detections with existing tracks
        pool_boxes = np.array([st.tlbr for st in strack_pool]) if strack_pool else np.empty((0, 4))
        det_boxes = np.array([d.tlbr for d in detections]) if detections else np.empty((0, 4))

        dists = 1.0 - compute_iou_batch(pool_boxes, det_boxes)
        matches_1, u_track_1, u_detection_1 = self._linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches_1:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 2: Match low-confidence detections with remaining tracked tracks
        r_tracked_stracks = [
            strack_pool[i] for i in u_track_1 if strack_pool[i].state == TrackState.Tracked
        ]
        r_pool_boxes = np.array([st.tlbr for st in r_tracked_stracks]) if r_tracked_stracks else np.empty((0, 4))
        det_boxes_second = np.array([d.tlbr for d in detections_second]) if detections_second else np.empty((0, 4))

        dists_2 = 1.0 - compute_iou_batch(r_pool_boxes, det_boxes_second)
        matches_2, u_track_2, _ = self._linear_assignment(dists_2, thresh=0.5)

        for itracked, idet in matches_2:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Unmatched tracks become Lost
        for it in u_track_2:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 3: Initialize new tracks for unmatched high-confidence detections
        for inew in u_detection_1:
            track = detections[inew]
            track.activate(self.frame_id)
            activated_stracks.append(track)

        # Step 4: Handle lost tracks and remove dead tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_lost_frames:
                track.mark_removed()
                removed_stracks.append(track)

        # Update tracked stracks
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = self._merge_tracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self._merge_tracks(self.tracked_stracks, refind_stracks)

        # Update lost stracks
        self.lost_stracks = [
            t for t in self.lost_stracks if t.state == TrackState.Lost
        ]
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = [
            t for t in self.lost_stracks if t not in self.tracked_stracks
        ]

        # Filter out tracks that are completely removed
        self.lost_stracks = [
            t for t in self.lost_stracks if (self.frame_id - t.frame_id) <= self.max_lost_frames
        ]
        self.removed_stracks.extend(removed_stracks)

        # Return all currently tracked active tracks
        return [t for t in self.tracked_stracks if t.is_activated]

    @staticmethod
    def _merge_tracks(track_list: List[STrack], new_tracks: List[STrack]) -> List[STrack]:
        track_dict: Dict[int, STrack] = {t.track_id: t for t in track_list}
        for t in new_tracks:
            track_dict[t.track_id] = t
        return list(track_dict.values())

    @staticmethod
    def _linear_assignment(
        cost_matrix: np.ndarray, thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian linear assignment with threshold gating."""
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: List[Tuple[int, int]] = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
                if r in unmatched_a:
                    unmatched_a.remove(r)
                if c in unmatched_b:
                    unmatched_b.remove(c)

        return matches, unmatched_a, unmatched_b
