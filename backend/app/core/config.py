import os
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure winget / local binary paths are included on Windows
if os.name == "nt":
    winget_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
    if os.path.isdir(winget_dir) and winget_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = winget_dir + os.pathsep + os.environ.get("PATH", "")


_CORE_DIR = Path(__file__).resolve().parent
_APP_DIR = _CORE_DIR.parent
_BACKEND_DIR = _APP_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
_ENV_FILE = _REPO_ROOT / ".env" if (_REPO_ROOT / ".env").exists() else _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_ENV_FILE), "../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    APP_VERSION: str = "1.1.0"
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
    CORS_ORIGINS: str = ""
    REDIS_URL: str | None = None
    REDIS_MAX_CONNECTIONS: int = 50
    PGBOUNCER_MODE: bool = False
    PROCESS_ROLE: str = "all"  # "all", "api", "worker"
    WORKER_MAX_CONCURRENCY: int = 1
    INFERENCE_DEVICE: str = "auto"  # "auto", "cuda", "cuda:0", "cpu"
    REQUIRE_GPU: bool = False
    SERVE_FRONTEND: bool | None = None
    METRICS_ENABLED: bool = True
    LOG_FORMAT: str = "json"
    STORAGE_BACKEND: str = "local"  # "local", "s3", "minio"
    S3_ENDPOINT_URL: str | None = None
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_REGION_NAME: str = "us-east-1"
    S3_BUCKET_NAME: str = "parallax-media"
    S3_FORCE_PATH_STYLE: bool = True
    S3_PRESIGNED_EXPIRATION_S: int = 3600

    # API Concurrency & Overload Protection (Workstream A)
    API_WORKERS: int = 4
    MAX_IN_FLIGHT_REQUESTS: int = 1500
    MAX_CONCURRENT_UPLOADS: int = 20

    # Database Connection Pool Budgeting (Workstream B)
    # Total connections = API_WORKERS * (DB_POOL_SIZE + DB_MAX_OVERFLOW) + Worker(5) + Admin(5)
    # 4 * (15 + 5) + 5 + 5 = 90 <= 100 (PostgreSQL max_connections)
    DB_POOL_SIZE: int = 15
    DB_MAX_OVERFLOW: int = 5
    DB_POOL_TIMEOUT: float = 10.0

    # Queue Backpressure & Controlled Rejection (Workstream C)
    ANALYSIS_QUEUE_MAX_DEPTH: int = 100
    ANALYSIS_QUEUE_MAX_WAITING_PER_TENANT: int = 5
    ANALYSIS_QUEUE_ADMISSION_TIMEOUT_S: float = 5.0

    # GPU-Aware Worker Scheduling (Workstream D)
    INFERENCE_MAX_CONCURRENT_JOBS: int = 1
    INFERENCE_GPU_MEMORY_RESERVE_MB: int = 1000
    INFERENCE_JOB_TIMEOUT_SECONDS: int = 120

    # Calibration thresholds from README §5
    P_FAKE_HI: float = 0.65
    P_FAKE_LO: float = 0.35

    @field_validator("SAMPLE_HOST_ALLOWLIST", mode="before")
    @classmethod
    def set_default_allowlist(cls, v: str | None) -> str:
        default = (
            "commons.wikimedia.org,upload.wikimedia.org,archive.org,"
            "ia800000.us.archive.org,thispersondoesnotexist.com,api.dicebear.com"
        )
        if not v or not str(v).strip():
            return default
        return str(v)

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
            self.COOKIE_SECURE = True
            if "*" in self.cors_origins_list:
                raise ValueError(
                    "Wildcard '*' CORS origin is not permitted in production with credentials"
                )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.CORS_ORIGINS:
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def should_serve_frontend(self) -> bool:
        if self.SERVE_FRONTEND is not None:
            return self.SERVE_FRONTEND
        return not self.is_production

    @property
    def storage_path(self) -> Path:
        p = Path(self.STORAGE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
