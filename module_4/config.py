"""
Module 4: Configuration and Data Contracts
Defines configuration parameters and data structures for Temporal Trajectory Gating
(Noise Filtering & ByteTrack) according to fyp_architecture_final_object.md Section 3 Module 4.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
import numpy as np


@dataclass
class Module4Config:
    """Configuration parameters for ByteTrack, Temporal Gating, and Keyframe Buffer."""

    # Frame Survival Gating (W consecutive frames required before confirming track)
    # Default 8 frames per fyp_architecture_final_object.md (MIN_TRACK_FRAMES = 8)
    survival_window: int = 8

    # Maximum frames a lost track is kept before being permanently removed (missed > 5 -> dispatch)
    max_lost_frames: int = 5

    # ByteTrack detection confidence threshold
    track_thresh: float = 0.50

    # IoU association matching threshold
    match_thresh: float = 0.70

    # Directional Persistence Check (Noise Filtering)
    directional_persistence: bool = True
    # Minimum net displacement / path ratio (MIN_DIR_PERSISTENCE = 0.50)
    min_dir_persistence: float = 0.50
    min_net_displacement: float = 10.0
    max_oscillation_ratio: float = 0.50

    # Active Learning Loop
    active_learning: bool = True
    active_learning_frame_thresh: int = 30
    active_learning_crop_dir: str = "outputs/active_learning_crops"
    active_learning_crop_padding: int = 10


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
    completed_track_payloads: List[Any] = field(default_factory=list)  # Emitted sharpest keyframes for Phase 2

    @property
    def confirmed_count(self) -> int:
        return len(self.confirmed_tracks)
