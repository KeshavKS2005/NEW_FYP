"""
Module 11 Configuration: Incremental YOLO Trainer & Champion-vs-Challenger Gate
Adheres strictly to Module 11 in fyp_architecture_final_object.md.
"""

from dataclasses import dataclass


@dataclass
class Module11Config:
    """Configuration parameters for incremental fine-tuning and metric evaluation gates."""

    # Fine-tuning hyperparameters
    epochs: int = 25
    imgsz: int = 640
    batch_size: int = 16
    freeze_layers: int = 10
    learning_rate: float = 0.001
    patience: int = 7

    # Metric evaluation gate thresholds
    max_old_map_drop: float = 0.015       # Gate 1: Max allowed drop in mAP50-95 on old classes (1.5%)
    min_new_class_map50: float = 0.60     # Gate 2: Min required mAP50 on newly discovered class
    min_new_class_recall: float = 0.65    # Gate 2: Min required Recall on newly discovered class

    # Runs output directory
    runs_dir: str = "models/runs"
