"""
Module 4: Configuration and Data Contracts
Defines configuration parameters and data structures for Temporal Trajectory Gating
(Noise Filtering & ByteTrack) according to fyp_architecture.md Section 3 Module 4 & Section 5.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class Module4Config:
    """Configuration parameters for ByteTrack, Temporal Gating, and Active Learning."""

    # Frame Survival Gating (W consecutive frames required before confirming track)
    # Kills 1-frame/transient glitches from camera auto-exposure or lighting changes
    survival_window: int = 5

    # Maximum frames a lost track is kept before being permanently removed
    max_lost_frames: int = 30

    # ByteTrack detection confidence threshold
    track_thresh: float = 0.50

    # IoU association matching threshold (higher = stricter overlap needed)
    match_thresh: float = 0.70

    # Directional Persistence Check (Noise Filtering)
    directional_persistence: bool = True
    # Minimum net displacement (pixels) required over track lifetime to reject stationary oscillations
    min_net_displacement: float = 15.0
    # Ratio of (net displacement / total path distance); if path length >= 10px and ratio < 0.30,
    # the object is oscillating back-and-forth in place (like a swaying tree branch or shadow)
    max_oscillation_ratio: float = 0.30

    # Active Learning Loop (fyp_architecture.md Section 5)
    # Triggers automatic cropping and dataset mining when an Unknown Track survives > 30 frames
    active_learning: bool = True
    active_learning_frame_thresh: int = 30
    active_learning_crop_dir: str = "outputs/active_learning_crops"
    active_learning_crop_padding: int = 10  # Pixel padding around crop


@dataclass
class ConfirmedUnknownTrack:
    """Represents a tracked unknown moving object."""

    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    age: int                     # Total frames since track creation
    hits: int                    # Total detection matches
    consecutive_hits: int        # Consecutive detection matches
    centroid: Tuple[int, int]
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    net_displacement: float = 0.0
    total_path_length: float = 0.0
    is_confirmed: bool = False   # True if survived >= survival_window & passed persistence
    is_noise: bool = False       # True if flagged as oscillating environmental noise

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


@dataclass
class Module4FrameResult:
    """Encapsulates output contracts for a processed frame from Module 4."""

    frame_index: int
    confirmed_tracks: List[ConfirmedUnknownTrack]   # Tracks that passed survival gating & persistence
    all_active_tracks: List[ConfirmedUnknownTrack]  # All tracked objects (including unconfirmed warm-ups)
    saved_crops: List[str]                          # File paths of crops saved by Active Learning hook
    processing_time_ms: float

    @property
    def confirmed_count(self) -> int:
        return len(self.confirmed_tracks)
