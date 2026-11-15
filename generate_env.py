
from pathlib import Path
import secrets

# project root directory
ROOT = Path(__file__).resolve().parent

# generate a secure random secret key
secret_key = secrets.token_urlsafe(48)

env_content = f"""# PARALLAX Environment Configuration

APP_ENV=development
SECRET_KEY={secret_key}

DATABASE_URL=sqlite:///./storage/parallax.db
STORAGE_DIR=./storage

ACCESS_TTL_MIN=15
REFRESH_TTL_DAYS=14
COOKIE_SECURE=false

MAX_UPLOAD_MB=200
MAX_DURATION_S=60

MODEL_PATH=./app/ml/weights/avt2_dwf.pt

FEED_CACHE_TTL_S=900
SAMPLE_HOST_ALLOWLIST=

MAX_CONCURRENT_JOBS=1
LOG_LEVEL=INFO
"""

env_path = ROOT / ".env"

# Never overwrite an existing environment file
if env_path.exists():
    print("ERROR: .env already exists!")
    print("Existing configuration was not modified.")
else:
    env_path.write_text(env_content, encoding="utf-8")
    print(".env file created successfully!")
    print("SECRET_KEY generated securely.")
    print(f"Location: {env_path}")
