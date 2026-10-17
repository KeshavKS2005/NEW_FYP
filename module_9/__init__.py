"""
Module 9: Automated VLM Class Discovery Trigger (Gemini 2.5 Flash)
"""

from module_9.config import Module9Config, get_gemini_api_key
from module_9.vlm_identifier import VLMDiscoveryIdentifier, GeminiObjectSchema

__all__ = ["Module9Config", "get_gemini_api_key", "VLMDiscoveryIdentifier", "GeminiObjectSchema"]
