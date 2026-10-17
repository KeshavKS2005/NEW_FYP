"""
Module 8: Relational Metadata & Observation Store (PostgreSQL with SQLite fallback)
"""

from module_8.config import Module8Config, load_phase1_env, build_postgres_dsn
from module_8.db_manager import DatabaseManager

__all__ = ["Module8Config", "DatabaseManager", "load_phase1_env", "build_postgres_dsn"]
