"""
Module 8: Backward-compatibility re-export of Module 9 VLMDiscoveryIdentifier
"""

from module_9.vlm_identifier import VLMDiscoveryIdentifier, GeminiObjectSchema
from module_9.config import Module9Config as Module8Config

__all__ = ["VLMDiscoveryIdentifier", "GeminiObjectSchema", "Module8Config"]
