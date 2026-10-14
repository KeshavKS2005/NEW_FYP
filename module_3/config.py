"""
Module 3: Configuration and Data Contracts
Defines configuration parameters and data structures for Mask-Level Erasure & Extraction
(The Discrepancy Engine) according to fyp_architecture.md Section 3 Module 3.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
import numpy as np


@dataclass
class Module3Config:
    """Configuration parameters for Mask-Level Erasure & Candidate Extraction."""

    # Dilation margin (padding percentage around YOLO known bounding box)
    # 0.10 corresponds to 10% box padding on all sides to cover limb swing / outer boundaries
    dilation_margin: float = 0.10

    # Minimum contour pixel area to accept as candidate unknown moving blob
    # User specified threshold: 100 px (rejects small noise, keeps distant objects)
    min_contour_area: float = 100.0

    # Hardcoded confidence assigned to surviving candidate boxes for ByteTrack
    tracker_confidence: float = 1.0

    # Optional minimum pixel padding on each side regardless of box percentage
    min_pad_px: int = 2

    # Morphological closing on residual mask to bridge fractured components of the same vehicle/object
    morph_close: bool = True
    morph_close_kernel_size: Tuple[int, int] = (11, 11)

    # Spatial proximity box merging (pixels):
    # If two candidate bounding boxes overlap or are within merge_proximity_px,
    # they are unified into a single bounding box before being passed to the tracker.
    merge_proximity_px: int = 20


@dataclass
class CandidateUnknownBox:
    """Represents a single candidate unknown moving bounding box extracted from residual motion."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    area: float = 0.0
    centroid: Tuple[int, int] = (0, 0)
    contour: Optional[Any] = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)

    def as_tracker_detection(self) -> List[float]:
        """Returns [x1, y1, x2, y2, confidence] format for tracker ingestion."""
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2), float(self.confidence)]


@dataclass
class Module3FrameResult:
    """Encapsulates output contracts for a processed frame from Module 3."""

    frame_index: int
    erased_mask: np.ndarray             # (H, W) uint8 binary mask with known objects erased (0)
    boxes: List[CandidateUnknownBox]    # Extracted candidate unknown bounding boxes
    box_matrix: np.ndarray              # Matrix [U, 5] formatted as [x1, y1, x2, y2, confidence]
    erased_known_count: int             # Number of known objects erased from the motion mask
    processing_time_ms: float           # Execution duration in milliseconds

    @property
    def candidate_count(self) -> int:
        return len(self.boxes)
