"""
Module 5: Compound Blob Decomposer (Method 2: FastSAM + DINOv2)
"""

from module_5.config import Module5Config, DecomposedSubObject
from module_5.compound_decomposer import CompoundBlobDecomposer
# Backward compatibility aliases
from module_4.keyframe_buffer import compute_sharpness, TrackKeyframeBuffer

__all__ = [
    "Module5Config",
    "DecomposedSubObject",
    "CompoundBlobDecomposer",
    "compute_sharpness",
    "TrackKeyframeBuffer"
]
