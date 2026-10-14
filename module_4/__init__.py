"""
Module 4: Temporal Trajectory Gating (Noise Filtering & Active Learning)
"""

from module_4.config import (
    Module4Config,
    ConfirmedUnknownTrack,
    Module4FrameResult,
)
from module_4.kalman_filter import KalmanBoxFilter
from module_4.byte_tracker import ByteTracker, STrack, TrackState
from module_4.temporal_gater import TemporalTrajectoryGater

__all__ = [
    "Module4Config",
    "ConfirmedUnknownTrack",
    "Module4FrameResult",
    "KalmanBoxFilter",
    "ByteTracker",
    "STrack",
    "TrackState",
    "TemporalTrajectoryGater",
]
