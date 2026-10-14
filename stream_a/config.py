"""
Stream A: Configuration and Data Contracts
Defines configuration parameters and data structures for class-agnostic motion extraction.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
import numpy as np


@dataclass
class MotionConfig:
    """Configuration parameters for OpenCV MOG2 and morphological filtering."""

    # MOG2 background subtractor parameters
    history: int = 500
    var_threshold: float = 16.0
    detect_shadows: bool = True

    # Shadow and foreground thresholding
    # MOG2 sets foreground to 255 and shadows to 127
    shadow_threshold: int = 250  # Values >= shadow_threshold are kept as foreground

    # Morphological filtering parameters
    morph_kernel_size: Tuple[int, int] = (5, 5)
    morph_open_iterations: int = 1   # Removes salt-and-pepper sensor noise and leaf flutter
    morph_close_iterations: int = 1  # Unifies fragmented moving components (e.g. limbs/torso)

    # Contour filtering
    min_contour_area: float = 200.0  # Discards tiny pixel clusters smaller than this area

    # Box Inflation / Padding Parameters (Enlarges boxes to cover full physical objects)
    box_pad_ratio: float = 0.25      # 25% inflation around detected motion
    box_pad_px: int = 10             # Minimum pixel expansion on every side
    merge_overlapping_boxes: bool = False  # Set to False so boxes stay completely separate and never combine

    # Learning rate (-1 = automatic based on history)
    learning_rate: float = -1.0


@dataclass
class MotionBoundingBox:
    """Represents a single detected moving component."""

    x1: int
    y1: int
    x2: int
    y2: int
    area: float
    centroid: Tuple[int, int]
    contour: Optional[Any] = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)


@dataclass
class MotionFrameResult:
    """Holds all motion artifacts for a single processed video frame."""

    frame_idx: int
    raw_mask: np.ndarray             # (H, W) uint8 from MOG2 (0=bg, 127=shadow, 255=fg)
    clean_mask: np.ndarray           # (H, W) uint8 binary mask strictly in {0, 255}
    boxes: List[MotionBoundingBox]   # Bounding boxes for all detected moving parts
    total_motion_pixels: int         # Count of active foreground pixels
    motion_ratio: float              # Fraction of frame covered by motion [0.0, 1.0]
