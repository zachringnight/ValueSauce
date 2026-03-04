"""FastAPI dependency injection for database and services."""

from __future__ import annotations

import logging
from functools import lru_cache

from ..config import get_settings, Settings
from ..utils.db import get_connection, Repository

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_repository() -> Repository:
    """Get a database repository instance."""
    settings = get_settings()
    conn = get_connection(settings.database_url)
    return Repository(conn)


def get_settings_dep() -> Settings:
    """FastAPI dependency for settings."""
    return get_settings()
