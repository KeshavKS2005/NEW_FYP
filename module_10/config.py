"""
Module 10 Configuration: Continual Dataset Builder & Teacher Pseudo-Labeler
Adheres strictly to Module 10 in fyp_architecture_final_object.md.
"""

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Module10Config:
    """Configuration parameters for dataset building, teacher pseudo-labeling, and replay."""

    # Base storage directory for incremental datasets
    datasets_base_dir: str = "storage/datasets"

    # Experience replay buffer directory
    replay_buffer_dir: str = "storage/replay_buffer"

    # Teacher model confidence threshold
    teacher_conf_threshold: float = 0.50

    # Train / Val split ratio
    train_split_ratio: float = 0.80

    # Number of sharpest samples to archive into replay buffer
    replay_archive_count: int = 20

    # Holdout class IDs excluded from teacher pseudo-labeling during testing
    holdout_class_ids: Set[int] = field(default_factory=set)
