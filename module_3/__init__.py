"""
Module 3: Mask-Level Erasure & Extraction (The Discrepancy Engine)
"""

from module_3.config import (
    Module3Config,
    CandidateUnknownBox,
    Module3FrameResult,
)
from module_3.erasure_engine import MaskErasureEngine

__all__ = [
    "Module3Config",
    "CandidateUnknownBox",
    "Module3FrameResult",
    "MaskErasureEngine",
]
