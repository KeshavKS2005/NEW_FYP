"""
Module 9 Configuration: Automated VLM Class Discovery Trigger (Gemini 2.5 Flash)
Adheres strictly to Module 9 in fyp_architecture_final_object.md.
"""

import os
from dataclasses import dataclass
from typing import Optional
from module_8.config import load_phase1_env


def get_gemini_api_key() -> Optional[str]:
    if os.environ.get("USE_MOCK_VLM", "0") == "1":
        return None
    load_phase1_env()
    return os.environ.get("GEMINI_API_KEY")


@dataclass
class Module9Config:
    """Configuration parameters for Gemini 2.5 Flash VLM identification agent."""

    # Model identifier (defaults to gemini-3.8-flash as validated with live API)
    model_name: str = os.environ.get("VISION_MODEL", "gemini-3.8-flash")

    # Gemini API Key loaded from FYP_phase1/.env
    api_key: Optional[str] = None

    # Observation count required to trigger VLM discovery
    trigger_count: int = int(os.environ.get("GEMINI_TRIGGER_COUNT", "100"))

    # Number of sharpest crop exemplars passed to multimodal prompt
    crop_exemplar_count: int = 3

    # LLM sampling temperature (0.1 for high factual determinism)
    temperature: float = 0.1

    # Minimum VLM confidence required to trigger Phase 3 Continual Retraining
    min_confidence_for_retrain: float = 0.70

    # Allow mock fallback for headless pytest / offline testing
    allow_mock_fallback: bool = True

    def __post_init__(self):
        if self.api_key is None and os.environ.get("USE_MOCK_VLM", "0") != "1":
            self.api_key = get_gemini_api_key()
