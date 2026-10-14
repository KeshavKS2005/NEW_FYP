"""
Stream B Package: Known-Object Semantic Detection
Implements Module 2 of fyp_architecture.md using Ultralytics YOLO.
"""

from stream_b.config import StreamBConfig, KnownBoundingBox, StreamBFrameResult
from stream_b.detector import YOLOKnownDetector
from stream_b.visualizer import StreamBVisualizer

__all__ = [
    "StreamBConfig",
    "KnownBoundingBox",
    "StreamBFrameResult",
    "YOLOKnownDetector",
    "StreamBVisualizer",
]
