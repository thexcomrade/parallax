import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure winget / local binary paths are included on Windows
if os.name == "nt":
    winget_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
    if os.path.isdir(winget_dir) and winget_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = winget_dir + os.pathsep + os.environ.get("PATH", "")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    SECRET_KEY: str = "insecure-dev-secret-key-change-in-production-min-32-chars"
    DATABASE_URL: str = "sqlite:///./storage/parallax.db"
    STORAGE_DIR: str = "./storage"
    ACCESS_TTL_MIN: int = 15
    REFRESH_TTL_DAYS: int = 7
    COOKIE_SECURE: bool = False
    MAX_UPLOAD_MB: int = 200
    MAX_DURATION_S: int = 60
    MODEL_PATH: str = "./app/ml/weights/avt2_dwf.pt"
    FEED_CACHE_TTL_S: int = 3600
    SAMPLE_HOST_ALLOWLIST: str = (
        "commons.wikimedia.org,upload.wikimedia.org,archive.org,"
        "ia800000.us.archive.org,thispersondoesnotexist.com,api.dicebear.com"
    )
    MAX_CONCURRENT_JOBS: int = 2
    LOG_LEVEL: str = "INFO"

    # Calibration thresholds from README §5
    P_FAKE_HI: float = 0.65
    P_FAKE_LO: float = 0.35

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.APP_ENV == "production":
            if (
                not self.SECRET_KEY
                or self.SECRET_KEY == "insecure-dev-secret-key-change-in-production-min-32-chars"
            ):
                raise ValueError("Production environment requires a secure non-default SECRET_KEY")
            if len(self.SECRET_KEY) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def storage_path(self) -> Path:
        p = Path(self.STORAGE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
