"""
Module 5: Keyframe Deduplication & Background Suppression Engine
Adheres to fyp_architecture_2.md Section 3 Module 5.

Key functionalities:
1. Sharpness evaluation via Laplacian variance on raw object crops.
2. Deduplication buffer: retains only the single sharpest keyframe per completed tracklet.
3. Background suppression: zeroes out static background pixels to neutral gray (128).
4. Square letterboxing: symmetrically pads crops to square format for DINOv2 input.
5. Disk persistence for contextual frames and cropped object patches.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image

from module_5.config import Module5Config


def compute_sharpness(crop_bgr: np.ndarray) -> float:
    """
    Computes image sharpness score using the variance of the Laplacian.
    Higher values indicate crisper edges and less motion blur.

    Args:
        crop_bgr: BGR image crop (H, W, 3).

    Returns:
        Sharpness score as a positive float (0.0 if empty or uniform).
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
    Maintains the highest quality (sharpest) observation for active unknown tracks.
    Ensures that each unique physical track emits exactly one keyframe observation.
    """

    def __init__(self, config: Optional[Module5Config] = None):
        self.config = config or Module5Config()
        # Maps track_id -> dict with keys: 'best_sharpness', 'payload'
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

        Args:
            track_id: Unique integer track ID.
            bbox: (x1, y1, x2, y2) bounding box coordinates.
            frame_bgr: Full BGR frame (H, W, 3).
            fg_mask: Residual binary motion mask (H, W).

        Returns:
            Computed sharpness for this frame observation.
        """
        x1, y1, x2, y2 = bbox
        H, W = frame_bgr.shape[:2]

        # Clip box to valid frame coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if (x2 - x1) < self.config.min_box_dim or (y2 - y1) < self.config.min_box_dim:
            return 0.0

        raw_crop = frame_bgr[y1:y2, x1:x2]
        mask_crop = fg_mask[y1:y2, x1:x2]

        sharpness = compute_sharpness(raw_crop)

        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "best_sharpness": -1.0,
                "payload": None
            }

        entry = self._tracks[track_id]
        if sharpness > entry["best_sharpness"]:
            entry["best_sharpness"] = sharpness
            entry["payload"] = (
                frame_bgr.copy(),
                raw_crop.copy(),
                mask_crop.copy(),
                (x1, y1, x2, y2),
                sharpness
            )

        return sharpness

    def get_best_payload(self, track_id: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int], float]]:
        """Returns the sharpest payload for track_id, if any."""
        entry = self._tracks.get(track_id)
        if entry is None or entry["payload"] is None:
            return None
        return entry["payload"]

    def pop_best_payload(self, track_id: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int], float]]:
        """Extracts and removes the sharpest payload when a track terminates."""
        entry = self._tracks.pop(track_id, None)
        if entry is None:
            return None
        return entry.get("payload")

    def clear(self):
        """Clears all stored track state."""
        self._tracks.clear()


class KeyframePreprocessor:
    """
    Processes keyframes for downstream feature extraction (DINOv2) and disk logging.
    """

    def __init__(self, config: Optional[Module5Config] = None):
        self.config = config or Module5Config()
        Path(self.config.frames_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.crops_dir).mkdir(parents=True, exist_ok=True)

    def save_disk_assets(
        self,
        track_id: int,
        frame_bgr: np.ndarray,
        raw_crop: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, str]:
        """
        Saves full context frame with red bounding box overlay and the raw crop to disk.

        Args:
            track_id: Integer track ID.
            frame_bgr: Full BGR frame.
            raw_crop: Cropped object patch in BGR.
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            Tuple of (frame_path, crop_path).
        """
        frame_vis = frame_bgr.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame_vis,
            f"DISCOVERY ID:{track_id}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        frame_path = os.path.join(self.config.frames_dir, f"track_{track_id:05d}.jpg")
        crop_path = os.path.join(self.config.crops_dir, f"track_{track_id:05d}.jpg")

        cv2.imwrite(frame_path, frame_vis)
        cv2.imwrite(crop_path, raw_crop)

        return frame_path, crop_path

    def preprocess_masked_square(
        self,
        raw_crop: np.ndarray,
        mask_crop: np.ndarray
    ) -> Image.Image:
        """
        Masks out background to neutral gray (128, 128, 128) and symmetrically pads to square.

        Args:
            raw_crop: (H, W, 3) BGR crop.
            mask_crop: (H, W) binary motion mask.

        Returns:
            Square RGB PIL Image with neutral gray background.
        """
        if raw_crop is None or raw_crop.size == 0:
            # Fallback for empty crop: return 224x224 gray image
            dim = self.config.target_dim
            return Image.new("RGB", (dim, dim), (self.config.bg_fill_value,) * 3)

        # 1. Mask out static background to neutral gray
        bg_gray = np.full_like(raw_crop, self.config.bg_fill_value)
        if mask_crop is not None and mask_crop.shape[:2] == raw_crop.shape[:2]:
            masked_crop = np.where(mask_crop[..., None] > 0, raw_crop, bg_gray)
        else:
            masked_crop = raw_crop

        # 2. Symmetrically pad to square
        h, w = masked_crop.shape[:2]
        dim = max(h, w)
        square_crop = np.full((dim, dim, 3), self.config.bg_fill_value, dtype=np.uint8)
        y_off = (dim - h) // 2
        x_off = (dim - w) // 2
        square_crop[y_off : y_off + h, x_off : x_off + w] = masked_crop

        # 3. Convert BGR to RGB PIL Image
        rgb_img = cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)
