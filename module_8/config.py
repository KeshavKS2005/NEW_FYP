"""
Module 8 Configuration: Relational Metadata & Observation Store (PostgreSQL)
Strictly reads database connection credentials from ../FYP_phase1/.env with zero code coupling.
Adheres strictly to fyp_architecture_final_object.md Section 2 & 7.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


def load_phase1_env() -> None:
    """Loads environment variables exclusively from sibling FYP_phase1/.env."""
    candidate_paths = [
        os.environ.get("FYP_PHASE1_ENV_PATH"),
        str(Path(__file__).resolve().parent.parent.parent / "FYP_phase1" / ".env"),
        str(Path(__file__).resolve().parent.parent.parent / "FYP_phase1" / "Sefl-Evolving-Video-Analytics-System" / ".env"),
    ]
    for env_path in candidate_paths:
        if env_path and os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)
            break


def build_postgres_dsn() -> str:
    """Builds PostgreSQL DSN from environment variables."""
    load_phase1_env()
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    host = os.environ.get("POSTGRES_HOST", os.environ.get("DB_HOST", "localhost"))
    port = os.environ.get("POSTGRES_PORT", os.environ.get("DB_PORT", "5432"))
    db = os.environ.get("POSTGRES_DB", os.environ.get("DB_NAME", "surveillance"))
    user = os.environ.get("POSTGRES_USER", os.environ.get("DB_USER", "postgres"))
    pwd = os.environ.get("POSTGRES_PASSWORD", os.environ.get("DB_PASS", ""))
    return f"dbname={db} user={user} password={pwd} host={host} port={port}"


@dataclass
class Module8Config:
    """Configuration parameters for database manager."""

    postgres_dsn: str = build_postgres_dsn()
    use_sqlite_fallback: bool = True
    sqlite_db_path: str = "outputs/surveillance_discovery.db"
    force_sqlite: bool = os.environ.get("USE_SQLITE", "0") == "1"

