"""Database connection and session management."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from engines.nil_props.config.settings import get_settings


def get_engine(url: str | None = None):
    settings = get_settings()
    db_url = url or settings.database_url
    engine = create_engine(db_url, echo=False)
    # Enable WAL mode for SQLite
    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def get_session(engine=None) -> Session:
    if engine is None:
        engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comment lines from a statement."""
    lines = []
    for line in sql.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            lines.append(line)
    return "\n".join(lines)


def init_db(engine=None):
    """Create all tables from schema.sql."""
    if engine is None:
        engine = get_engine()
    schema_path = Path(__file__).parent.parent / "sql" / "schema.sql"
    sql = schema_path.read_text()
    with engine.begin() as conn:
        for statement in sql.split(";"):
            stmt = _strip_sql_comments(statement).strip()
            if stmt:
                conn.execute(text(stmt))
    return engine
