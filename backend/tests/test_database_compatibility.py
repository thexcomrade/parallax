"""Tests for SQLite and PostgreSQL database compatibility, normalization, and connection pooling."""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Analysis, AnalysisArtifact, Base, User
from app.db.session import normalize_database_url


def test_database_url_normalization() -> None:
    """PostgreSQL URLs must be normalized to use psycopg v3 driver."""
    assert normalize_database_url("postgres://user:pass@host:5432/db") == "postgresql+psycopg://user:pass@host:5432/db"
    assert normalize_database_url("postgresql://user:pass@host:5432/db") == "postgresql+psycopg://user:pass@host:5432/db"
    assert normalize_database_url("postgresql+psycopg://user:pass@host:5432/db") == "postgresql+psycopg://user:pass@host:5432/db"
    assert normalize_database_url("sqlite:///./parallax.db") == "sqlite:///./parallax.db"


def test_sqlite_in_memory_crud_and_foreign_keys() -> None:
    """Verify ORM models, relationships, and cascade deletes work cleanly."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    with TestingSession() as session:
        # Create user
        user = User(
            email="compat_test@parallax.io",
            password_hash="hash_placeholder",
            role="operator",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        assert user.id is not None

        # Create analysis
        analysis = Analysis(
            id="test-compat-analysis-uuid",
            user_id=user.id,
            original_name="clip.mp4",
            sha256="1" * 64,
            size_bytes=1024,
            duration_s=2.5,
            status="completed",
            stage="done",
            verdict="REAL",
            audio_present=True,
            model_version="1.0.0",
            model_mode="trained",
        )
        session.add(analysis)
        session.commit()

        # Create artifact
        art = AnalysisArtifact(
            analysis_id=analysis.id,
            kind="heatmap",
            name="test_heatmap.png",
            path="/tmp/test_heatmap.png",
        )
        session.add(art)
        session.commit()

        # Verify query and relationship
        loaded = session.execute(select(Analysis).where(Analysis.id == analysis.id)).scalar_one()
        assert len(loaded.artifacts) == 1
        assert loaded.artifacts[0].name == "test_heatmap.png"

        # Verify cascade deletion
        session.delete(user)
        session.commit()

        analysis_stmt = select(Analysis).where(Analysis.id == analysis.id)
        assert session.execute(analysis_stmt).scalar_one_or_none() is None
        art_stmt = select(AnalysisArtifact).where(
            AnalysisArtifact.name == "test_heatmap.png"
        )
        assert session.execute(art_stmt).scalar_one_or_none() is None


def test_postgres_engine_configuration_parameters() -> None:
    """Verify PostgreSQL engine initializes with correct pool parameters without connecting."""
    pg_url = "postgresql+psycopg://fake_user:fake_pass@localhost:5432/fake_db"
    normalized = normalize_database_url(pg_url)

    # Engine initialization should configure pool size and pre-ping
    engine = create_engine(
        normalized,
        pool_size=10,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    assert engine.dialect.name == "postgresql"
    assert engine.pool.size() == 10
