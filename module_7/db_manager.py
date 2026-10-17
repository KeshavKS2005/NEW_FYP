"""
Module 7: Relational Database Manager (Alias to Module 8 DatabaseManager)
Maintained for backward compatibility with 8-module imports.
"""

from module_8.db_manager import DatabaseManager
from module_8.config import Module8Config as Module7Config

__all__ = ["DatabaseManager", "Module7Config"]
