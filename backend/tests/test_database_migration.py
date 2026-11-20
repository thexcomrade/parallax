"""Tests for database schema migration safety.

Verifies that upgrading an existing database containing users succeeds,
preserves existing data, and assigns pre-existing users the safe 'operator' role.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.models import User


@pytest.fixture
def legacy_sqlite_db() -> str:
    """Create a temporary legacy SQLite database without the 'role' column."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Create legacy users table (v1 schema)
    cur.execute(
        """
        CREATE TABLE users (
            id VARCHAR(36) PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at DATETIME NOT NULL
        )
        """
    )
    # Insert legacy test users
    cur.execute(
        """
        INSERT INTO users (id, email, password_hash, created_at)
        VALUES
            ('legacy-user-1', 'admin@example.com', 'hash1', '2026-01-01 00:00:00'),
            ('legacy-user-2', 'analyst@example.com', 'hash2', '2026-01-02 00:00:00')
        """
    )
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    p = Path(db_path)
    if p.exists():
        p.unlink()


def test_legacy_database_migration_adds_role_column(legacy_sqlite_db: str) -> None:
    """Verify that adding the role column preserves existing rows with 'operator' role."""
    conn = sqlite3.connect(legacy_sqlite_db)
    cur = conn.cursor()

    # Pre-check: 'role' should not exist
    cur.execute("PRAGMA table_info(users)")
    cols = [col[1] for col in cur.fetchall()]
    assert "role" not in cols

    # Apply migration: ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'operator'
    cur.execute(
        "ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'operator'"
    )
    conn.commit()

    # Post-check: 'role' column now exists
    cur.execute("PRAGMA table_info(users)")
    cols_after = [col[1] for col in cur.fetchall()]
    assert "role" in cols_after

    # Verify existing rows received 'operator' role
    cur.execute("SELECT id, email, role FROM users ORDER BY id")
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0] == ("legacy-user-1", "admin@example.com", "operator")
    assert rows[1] == ("legacy-user-2", "analyst@example.com", "operator")

    conn.close()


def test_migrated_database_compatible_with_sqlalchemy_orm(legacy_sqlite_db: str) -> None:
    """Verify SQLAlchemy ORM can query and manipulate users from the migrated database."""
    conn = sqlite3.connect(legacy_sqlite_db)
    cur = conn.cursor()
    cur.execute(
        "ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'operator'"
    )
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{legacy_sqlite_db}")
    with Session(engine) as session:
        # Query existing legacy users
        users = session.scalars(select(User).order_by(User.id)).all()
        assert len(users) == 2
        assert users[0].role == "operator"
        assert users[1].role == "operator"

        # Insert new user via ORM without role -> should default to 'public'
        new_user = User(
            id="new-public-user",
            email="newuser@example.com",
            password_hash="newhash",
        )
        session.add(new_user)
        session.commit()

        # Query newly added user
        retrieved = session.get(User, "new-public-user")
        assert retrieved is not None
        assert retrieved.role == "public"

    engine.dispose()
