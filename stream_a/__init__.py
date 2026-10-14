"""
Stream A Package: Class-Agnostic Motion Extraction
"""

from stream_a.config import MotionConfig, MotionBoundingBox, MotionFrameResult
from stream_a.motion_extractor import MOG2MotionExtractor
from stream_a.visualizer import StreamAVisualizer

__all__ = [
    "MotionConfig",
    "MotionBoundingBox",
    "MotionFrameResult",
    "MOG2MotionExtractor",
    "StreamAVisualizer",
]
