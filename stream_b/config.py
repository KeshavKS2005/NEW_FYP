"""
Stream B: Configuration and Data Contracts
Defines configuration parameters and data structures for known-object semantic detection.
Adheres strictly to Module 2 specifications in fyp_architecture_final_object.md.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Set
import numpy as np


@dataclass
class StreamBConfig:
    """Configuration parameters for YOLO Known-Object Detector (Stream B)."""

    # Model weights path (defaults to production champion, bootstrapped from yolo11n.pt)
    model_path: str = "models/champion_latest.pt"

    # Inference parameters
    conf_threshold: float = 0.35    # Low-confidence threshold filter (< 0.35 discarded)
    iou_threshold: float = 0.45     # NMS IoU threshold to deduplicate detections
    imgsz: int = 640                # Inference image dimension

    # Hardware target: '0' for CUDA GPU, 'cpu' for CPU fallback
    device: str = "cpu"

    # Known class catalog (derived from model.names or customized)
    known_classes: List[str] = field(default_factory=lambda: ["bicycle"])

    # Class IDs to hold out as unknowns for testing (e.g. [0, 1] for Person, Bicycle)
    holdout_class_ids: Set[int] = field(default_factory=set)


@dataclass
class KnownBoundingBox:
    """Represents a single detected known object."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    class_name: str = "known_object"

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
