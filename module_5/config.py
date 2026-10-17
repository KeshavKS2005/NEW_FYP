"""
Module 5: Configuration and Data Contracts
Defines configuration parameters and data structures for Compound Blob Decomposition
(Method 2: FastSAM + DINOv2 Iterative Cosine Merging & MaskCut Fallback).
Adheres strictly to Module 5 specifications in fyp_architecture_final_object.md.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
import numpy as np


@dataclass
class Module5Config:
    """Configuration parameters for Compound Blob Decomposer (Module 5)."""

    fastsam_weights: str = "FastSAM-s.pt"
    min_sub_object_area: int = 350
    part_min_fg_ratio: float = 0.06
    part_merge_cosine_thresh: float = 0.62
    maskcut_sep_thresh: float = 0.22
    device: str = "cpu"


@dataclass
class DecomposedSubObject:
    """Represents a single isolated sub-object produced by decomposition."""

    bbox: Tuple[int, int, int, int]    # [x1, y1, x2, y2] in full-frame coordinates
    raw_crop: np.ndarray               # Sub-object raw BGR crop
    sub_mask: np.ndarray               # Sub-object isolated binary mask
    method: str                        # "METHOD2_FASTSAM_DINOV2", "FALLBACK_MASKCUT", or "SINGLE"
