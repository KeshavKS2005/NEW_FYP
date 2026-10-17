"""
Module 12 Configuration: Atomic Model Registry & Zero-Downtime Hot-Swapper
Adheres strictly to Module 12 in fyp_architecture_final_object.md.
"""

from dataclasses import dataclass


@dataclass
class Module12Config:
    """Configuration parameters for atomic hot-swap deployer."""

    champion_weights_path: str = "models/champion_latest.pt"
    staging_weights_path: str = "models/champion_staging.pt"
