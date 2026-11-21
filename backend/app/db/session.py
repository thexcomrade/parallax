from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

__all__ = ["Base", "SessionLocal", "engine", "get_db", "normalize_database_url"]


class Base(DeclarativeBase):
    """Typed declarative base for all ORM models."""


def normalize_database_url(url: str) -> str:
    """Normalize PostgreSQL URLs to ensure psycopg v3 driver is used with SQLAlchemy."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


db_url = normalize_database_url(settings.DATABASE_URL)
connect_args: dict[str, Any] = {}
engine_kwargs: dict[str, Any] = {
    "pool_pre_ping": True,
}

if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False
else:
    # PostgreSQL pool configuration budgeted for multi-worker scaling:
    # API_WORKERS(4) * (15 + 5) + Worker(5) + Admin(5) = 90 <= 100 max_connections
    engine_kwargs["pool_size"] = settings.DB_POOL_SIZE
    engine_kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW
    engine_kwargs["pool_timeout"] = settings.DB_POOL_TIMEOUT
    engine_kwargs["pool_recycle"] = 300
    if settings.PGBOUNCER_MODE:
        connect_args["prepare_threshold"] = None

engine = create_engine(
    db_url,
    connect_args=connect_args,
    **engine_kwargs,
)


# SQLite PRAGMA enforcement for WAL mode and Foreign Keys
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
    if settings.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session and safely closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
