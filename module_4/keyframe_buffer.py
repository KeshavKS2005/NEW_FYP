"""
Module 4: Track Keyframe Buffer
Maintains the highest quality (sharpest) observation for active unknown tracks.
Ensures that each unique physical track emits exactly one keyframe observation.
Adheres strictly to Module 4 specifications in fyp_architecture_final_object.md.
"""

from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np


def compute_sharpness(crop_bgr: np.ndarray) -> float:
    """
    Computes image sharpness score using the variance of the Laplacian.
    Higher values indicate crisper edges and less motion blur.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr

    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


class TrackKeyframeBuffer:
    """
    Maintains the single sharpest keyframe observation per unknown track.
    Emits payload when the track is confirmed and terminates.
    """

    def __init__(self, min_box_dim: int = 16):
        self.min_box_dim = min_box_dim
        # Maps track_id -> dict with keys: 'best_sharpness', 'payload', 'dispatched'
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        frame_bgr: np.ndarray,
        fg_mask: np.ndarray
    ) -> float:
        """
        Updates the track buffer with a new frame observation.
        Retains unedited frame_bgr, raw crop, and mask crop if sharpness is highest.
        """
        x1, y1, x2, y2 = bbox
        H, W = frame_bgr.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if (x2 - x1) < self.min_box_dim or (y2 - y1) < self.min_box_dim:
            return 0.0

        raw_crop = frame_bgr[y1:y2, x1:x2]
        mask_crop = fg_mask[y1:y2, x1:x2]

        sharpness = compute_sharpness(raw_crop)

        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "best_sharpness": -1.0,
                "payload": None,
                "dispatched": False,
            }

        entry = self._tracks[track_id]
        if sharpness > entry["best_sharpness"]:
            entry["best_sharpness"] = sharpness
            # Clean unedited frame_bgr copy
            entry["payload"] = (
                frame_bgr.copy(),
                raw_crop.copy(),
                mask_crop.copy(),
                (x1, y1, x2, y2),
                sharpness
            )

        return sharpness

    def get_best_payload(self, track_id: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int], float]]:
        entry = self._tracks.get(track_id)
        if entry is not None and entry["payload"] is not None:
            return entry["payload"]
        return None

    def mark_dispatched(self, track_id: int):
        if track_id in self._tracks:
            self._tracks[track_id]["dispatched"] = True

    def is_dispatched(self, track_id: int) -> bool:
        return self._tracks.get(track_id, {}).get("dispatched", False)

    def pop_best_payload(self, track_id: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int], float]]:
        entry = self._tracks.pop(track_id, None)
        if entry is not None:
            return entry["payload"]
        return None

    def clear(self):
        self._tracks.clear()
