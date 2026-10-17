"""
Pytest global configuration and fixtures.
Ensures unit and integration test suites run in hermetic headless isolation using
temporary SQLite databases and deterministic mock VLM responses.
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_test_isolation():
    """Sets isolated environment variables for the entire pytest session."""
    original_sqlite = os.environ.get("USE_SQLITE")
    original_vlm = os.environ.get("USE_MOCK_VLM")

    os.environ["USE_SQLITE"] = "1"
    os.environ["USE_MOCK_VLM"] = "1"

    yield

    if original_sqlite is not None:
        os.environ["USE_SQLITE"] = original_sqlite
    else:
        os.environ.pop("USE_SQLITE", None)

    if original_vlm is not None:
        os.environ["USE_MOCK_VLM"] = original_vlm
    else:
        os.environ.pop("USE_MOCK_VLM", None)
