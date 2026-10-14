"""
Stream B: Known-Object Detector (Module 2)
Implements YOLO known object detection according to fyp_architecture.md:
- Identifies bounding boxes of known objects in current frame.
- Applies Non-Maximum Suppression (NMS).
- Filters out detections below confidence threshold (< 0.40).
- Produces coordinate matrix [K, 4] formatted as (x1, y1, x2, y2).
"""

import time
from typing import Optional, List
import numpy as np
import torch
from ultralytics import YOLO

from stream_b.config import StreamBConfig, KnownBoundingBox, StreamBFrameResult


class YOLOKnownDetector:
    """Ultralytics YOLO Known-Object Detector for Stream B Semantics."""

    def __init__(self, config: Optional[StreamBConfig] = None):
        """
        Initializes the YOLO detector with configured weights and threshold.
        """
        self.config = config or StreamBConfig()

        # Check device availability
        if self.config.device != "cpu" and not torch.cuda.is_available():
            print("[WARN] CUDA not available in current process. Falling back to CPU for Stream B.")
            self.device = "cpu"
        else:
            self.device = self.config.device

        print(f"[Stream B] Loading model from: {self.config.model_path} (device={self.device})")
        self.model = YOLO(self.config.model_path)
        print(f"[Stream B] Model loaded. Known classes: {self.config.known_classes}")

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> StreamBFrameResult:
        """
        Processes a single RGB frame and extracts known-object bounding boxes.

        Args:
            frame: Raw BGR/RGB frame (H, W, 3).
            frame_index: Sequential integer index of the frame.

        Returns:
            StreamBFrameResult containing bounding box objects and [K, 4] numpy matrix.
        """
        if frame is None or frame.size == 0:
            return StreamBFrameResult(
                frame_index=frame_index,
                boxes=[],
                box_matrix=np.empty((0, 4), dtype=np.int32),
                inference_time_ms=0.0
            )

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        # YOLO inference with NMS and confidence thresholding
        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False
        )

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        boxes_list: List[KnownBoundingBox] = []
        matrix_rows = []

        if results and len(results) > 0 and results[0].boxes is not None:
            raw_boxes = results[0].boxes
            for i in range(len(raw_boxes)):
                xyxy = raw_boxes.xyxy[i].cpu().numpy()
                conf = float(raw_boxes.conf[i].cpu().item())
                cls_id = int(raw_boxes.cls[i].cpu().item())

                # Strict confidence threshold verification
                if conf < self.config.conf_threshold:
                    continue

                x1 = int(np.clip(np.round(xyxy[0]), 0, w))
                y1 = int(np.clip(np.round(xyxy[1]), 0, h))
                x2 = int(np.clip(np.round(xyxy[2]), 0, w))
                y2 = int(np.clip(np.round(xyxy[3]), 0, h))

                # Ensure valid box dimensions
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_name = (
                    self.config.known_classes[cls_id]
                    if cls_id < len(self.config.known_classes)
                    else "known_object"
                )

                box = KnownBoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                )
                boxes_list.append(box)
                matrix_rows.append([x1, y1, x2, y2])

        if len(matrix_rows) > 0:
            box_matrix = np.array(matrix_rows, dtype=np.int32)
        else:
            box_matrix = np.empty((0, 4), dtype=np.int32)

        return StreamBFrameResult(
            frame_index=frame_index,
            boxes=boxes_list,
            box_matrix=box_matrix,
            inference_time_ms=inference_time_ms
        )
