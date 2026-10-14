"""
Stream A: Class-Agnostic Motion Extractor
Implements OpenCV MOG2 background subtraction with shadow suppression,
morphological noise reduction, and moving component bounding box extraction.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from stream_a.config import MotionConfig, MotionBoundingBox, MotionFrameResult


class MOG2MotionExtractor:
    """
    Extracts all moving pixels and moving parts from video frames regardless of class.
    
    Adheres to Module 1 of the Dual-Stream Motion-YOLO Discrepancy Pipeline:
    1. OpenCV MOG2 Gaussian Mixture Model for background subtraction with shadow tracking.
    2. Explicit thresholding to drop shadow pixels (value 127) and retain pure foreground (255).
    3. Morphological opening (5x5 kernel) to strip salt-and-pepper noise and leaf flutter.
    4. Morphological closing to bridge fractured moving parts (e.g. limbs/torso).
    5. Contour extraction and bounding box generation for all moving components.
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()

        # Initialize OpenCV MOG2 background subtractor
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows
        )

        # Precompute morphological structuring element
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.morph_kernel_size
        )

        self._frame_count = 0

    def reset(self) -> None:
        """Resets the background model and frame counter."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows
        )
        self._frame_count = 0

    def _merge_overlapping_boxes(self, boxes: List[MotionBoundingBox]) -> List[MotionBoundingBox]:
        """Merges boxes that overlap or touch so single objects are unified."""
        if len(boxes) <= 1:
            return boxes

        merged = []
        unmerged = list(boxes)

        while unmerged:
            curr = unmerged.pop(0)
            has_merged = False
            for idx, other in enumerate(unmerged):
                # Check for overlap or touch
                if not (curr.x2 < other.x1 or curr.x1 > other.x2 or
                        curr.y2 < other.y1 or curr.y1 > other.y2):
                    x1 = min(curr.x1, other.x1)
                    y1 = min(curr.y1, other.y1)
                    x2 = max(curr.x2, other.x2)
                    y2 = max(curr.y2, other.y2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    new_box = MotionBoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        area=float((x2 - x1) * (y2 - y1)),
                        centroid=(cx, cy)
                    )
                    unmerged.pop(idx)
                    unmerged.append(new_box)
                    has_merged = True
                    break
            if not has_merged:
                merged.append(curr)

        return merged

    def process(self, frame: np.ndarray) -> MotionFrameResult:
        """
        Processes a single video frame.

        Args:
            frame: (H, W, 3) BGR or RGB image array.

        Returns:
            MotionFrameResult with raw MOG2 mask, cleaned binary mask, and detected boxes.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid empty frame passed to motion extractor.")

        self._frame_count += 1
        h, w = frame.shape[:2]

        # 1. MOG2 Background Subtraction
        # Output values: 0 (background), 127 (shadow), 255 (foreground)
        raw_mask = self.subtractor.apply(frame, learningRate=self.config.learning_rate)

        # 2. Thresholding: drop shadows (127) and background (0), strictly keep foreground (255)
        _, clean_mask = cv2.threshold(
            raw_mask,
            self.config.shadow_threshold - 1,
            255,
            cv2.THRESH_BINARY
        )

        # 3. Morphological Opening (Erosion followed by Dilation) to strip noise
        if self.config.morph_open_iterations > 0:
            clean_mask = cv2.morphologyEx(
                clean_mask,
                cv2.MORPH_OPEN,
                self.kernel,
                iterations=self.config.morph_open_iterations
            )

        # 4. Morphological Closing (Dilation followed by Erosion) to unify fragmented parts
        if self.config.morph_close_iterations > 0:
            clean_mask = cv2.morphologyEx(
                clean_mask,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=self.config.morph_close_iterations
            )

        # 5. Extract moving contours and bounding boxes
        contours, _ = cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[MotionBoundingBox] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.config.min_contour_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = x + bw // 2
                cy = y + bh // 2

            # Inflate / enlarge the bounding box to envelope the physical moving object
            pad_x = max(self.config.box_pad_px, int(bw * self.config.box_pad_ratio))
            pad_y = max(self.config.box_pad_px, int(bh * self.config.box_pad_ratio))
            bx1 = max(0, x - pad_x)
            by1 = max(0, y - pad_y)
            bx2 = min(w, x + bw + pad_x)
            by2 = min(h, y + bh + pad_y)

            boxes.append(MotionBoundingBox(
                x1=bx1,
                y1=by1,
                x2=bx2,
                y2=by2,
                area=float((bx2 - bx1) * (by2 - by1)),
                centroid=(cx, cy),
                contour=cnt
            ))

        # Merge overlapping/touching padded boxes belonging to the same object
        if self.config.merge_overlapping_boxes and len(boxes) > 1:
            boxes = self._merge_overlapping_boxes(boxes)

        # Sort boxes by area descending for consistent downstream processing
        boxes.sort(key=lambda b: b.area, reverse=True)

        total_motion_pixels = int(cv2.countNonZero(clean_mask))
        motion_ratio = float(total_motion_pixels) / float(h * w) if (h * w) > 0 else 0.0

        return MotionFrameResult(
            frame_idx=self._frame_count,
            raw_mask=raw_mask,
            clean_mask=clean_mask,
            boxes=boxes,
            total_motion_pixels=total_motion_pixels,
            motion_ratio=motion_ratio
        )
