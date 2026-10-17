"""
Stream B: Known-Object Detector (Module 2)
Implements YOLO known object detection according to fyp_architecture_final_object.md:
- Identifies standard bounding boxes [x1, y1, x2, y2] of known objects in current frame.
- Applies Non-Maximum Suppression (NMS).
- Filters out detections below confidence threshold (< 0.35) and holdout classes.
- Produces coordinate matrix [K, 4] formatted as (x1, y1, x2, y2).
- Supports live weight reloading in-place (hot-swap zero downtime).
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional, List, Set
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
            self.device = "cpu"
        else:
            self.device = self.config.device

        # Bootstrap champion weights if not present
        weights_path = Path(self.config.model_path)
        if not weights_path.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            fallback = Path("yolo11n.pt")
            if fallback.exists():
                shutil.copy(fallback, weights_path)
            else:
                YOLO("yolo11n.pt")
                if Path("yolo11n.pt").exists():
                    shutil.copy("yolo11n.pt", weights_path)

        self._load_model(str(weights_path))

    def _load_model(self, model_path: str):
        self.model = YOLO(model_path)
        if hasattr(self.model, "names") and self.model.names:
            self.class_names = dict(self.model.names)
        else:
            self.class_names = {i: name for i, name in enumerate(self.config.known_classes)}

    def reload_weights(self, weights_path: str):
        """Atomically reloads model weights in-place without dropping frames."""
        print(f"[Stream B] Hot-swapping YOLO weights to: {weights_path}")
        self._load_model(weights_path)
        # Clear holdout class IDs when retrained model is deployed
        self.config.holdout_class_ids.clear()
        print(f"[Stream B] Weights hot-swapped successfully. Active classes: {len(self.class_names)}")

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> StreamBFrameResult:
        """
        Processes a single RGB/BGR frame and extracts known-object bounding boxes.

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

        # YOLO standard bounding box inference
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

                # Skip holdout class IDs during testing
                if cls_id in self.config.holdout_class_ids:
                    continue

                if conf < self.config.conf_threshold:
                    continue

                x1 = int(np.clip(np.round(xyxy[0]), 0, w))
                y1 = int(np.clip(np.round(xyxy[1]), 0, h))
                x2 = int(np.clip(np.round(xyxy[2]), 0, w))
                y2 = int(np.clip(np.round(xyxy[3]), 0, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                cls_name = self.class_names.get(cls_id, "known_object")

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
