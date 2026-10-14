"""
Stream B: Configuration and Data Contracts
Defines configuration parameters and data structures for known-object semantic detection.
Adheres strictly to Module 2 specifications in fyp_architecture.md.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class StreamBConfig:
    """Configuration parameters for YOLO Known-Object Detector (Stream B)."""

    # Model weights path
    model_path: str = "stream_b/weights/known_single_object.pt"

    # Inference parameters (per fyp_architecture.md Section 3 Module 2)
    conf_threshold: float = 0.40    # Low-confidence threshold filter (< 0.40 discarded)
    iou_threshold: float = 0.45     # NMS IoU threshold to deduplicate detections
    imgsz: int = 640                # Inference image dimension

    # Hardware target: '0' for CUDA GPU, 'cpu' for CPU fallback
    device: str = "0"

    # Known class catalog (strictly single object, non-person and non-car)
    known_classes: List[str] = field(default_factory=lambda: ["bicycle"])


@dataclass
class KnownBoundingBox:
    """Represents a single detected known object."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    class_name: str = "bicycle"

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    @property
    def centroid(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class StreamBFrameResult:
    """Encapsulates output contracts for a processed frame from Stream B."""

    frame_index: int
    boxes: List[KnownBoundingBox]
    # Matrix of coordinates [K, 4] formatted as (x1, y1, x2, y2)
    box_matrix: np.ndarray
    inference_time_ms: float

    @property
    def known_count(self) -> int:
        return len(self.boxes)
