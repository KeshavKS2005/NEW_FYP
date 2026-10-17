"""
Module 6 Configuration: Self-Supervised Feature Extraction via DINOv2 with Registers
"""

from dataclasses import dataclass
import torch


@dataclass
class Module6Config:
    """Configuration parameters for DINOv2 feature extractor."""

    # Model architecture identifier in PyTorch Hub
    model_name: str = "dinov2_vits14_reg"

    # Hub repository
    repo_name: str = "facebookresearch/dinov2"

    # Fallback HuggingFace repo if hub lookup encounters network issue
    hf_fallback_repo: str = "facebook/dinov2-with-registers-small"

    # Output feature embedding dimension
    embedding_dim: int = 384

    # Target input spatial resolution
    image_size: int = 224

    # Compute device ('cuda', 'cpu')
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
