import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

# Set test environment variables before importing app
temp_dir = tempfile.mkdtemp()
test_db_path = Path(temp_dir) / "test_parallax.db"
os.environ["APP_ENV"] = "development"
os.environ["TESTING"] = "1"
os.environ["SECRET_KEY"] = "test-secret-key-that-is-at-least-32-chars-long-for-testing"
os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"
os.environ["STORAGE_DIR"] = str(Path(temp_dir) / "storage")

from app.core.deps import get_db  # noqa: E402
from app.db.models import Base, User  # noqa: E402
from app.db.session import engine as app_engine  # noqa: E402
from app.main import app  # noqa: E402

# Create tables in test db
Base.metadata.create_all(bind=app_engine)


@pytest.fixture(scope="session")
def db_engine() -> Any:
    return app_engine


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a transactional database session rolled back after each test."""
    connection = app_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, autocommit=False, autoflush=False)()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Test client with overridden database dependency."""

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, base_url="http://testserver") as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session: Session) -> User:
    """Fixture providing a standard pre-registered test user."""
    from app.services.auth_service import AuthService

    user = AuthService.register_user(
        db=db_session,
        email="forensics_analyst@parallax.io",
        password="SecureMasterPassword2026!",
        role="operator",
    )
    return user


@pytest.fixture(scope="session")
def synthetic_video_with_audio() -> Path:
    """Generate a tiny 3-second test clip with sine tone and moving testsrc."""
    import subprocess

    from app.ml.preprocess.audio import _get_ffmpeg_bin

    ffmpeg_bin = _get_ffmpeg_bin()
    clip_path = Path(temp_dir) / "synthetic_av.mp4"
    if not clip_path.exists():
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(clip_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=20.0)
    return clip_path


@pytest.fixture(scope="session")
def synthetic_video_no_audio() -> Path:
    """Generate a 3-second video clip with NO audio stream."""
    import subprocess

    from app.ml.preprocess.audio import _get_ffmpeg_bin

    ffmpeg_bin = _get_ffmpeg_bin()
    clip_path = Path(temp_dir) / "synthetic_no_audio.mp4"
    if not clip_path.exists():
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=20.0)
    return clip_path


@pytest.fixture(scope="session")
def corrupt_file() -> Path:
    """Generate an unreadable corrupted media file."""
    bad_path = Path(temp_dir) / "corrupt_video.mp4"
    bad_path.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd\xfcNOT_A_REAL_VIDEO_HEADER")
    return bad_path
