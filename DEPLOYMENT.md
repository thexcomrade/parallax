# PARALLAX — Production Deployment Runbook

This document details the production deployment, containerization, environment configuration, and rollback procedures for PARALLAX (Audio-Visual Deepfake Detection System).

---

## 1. Prerequisites

- **Host Operating System**: Linux (Ubuntu 22.04 LTS recommended) or Container Host.
- **Runtimes**:
  - Docker 24.0+ and Docker Compose v2.
  - FFmpeg 6.0+ with libvpx, libx264, and libmp3lame support.
  - Python 3.12 (if deploying directly to host).
  - Node.js 20 LTS (if building frontend manually).
- **Hardware Recommendations**:
  - CPU: Minimum 4 vCPUs (x86_64 or arm64).
  - RAM: Minimum 8 GB (for PyTorch tensor processing and FFmpeg frame pipelines).
  - Storage: Minimum 50 GB NVMe/SSD for uploads and forensic artifacts.

---

## 2. Environment Configuration

Create a production `.env` file in the repository root by copying `.env.example`:

```bash
cp .env.example .env
chmod 600 .env
```

Populate the required environment variables:

| Variable | Description | Production Example |
|---|---|---|
| `APP_ENV` | Application environment mode | `production` |
| `SECRET_KEY` | 64+ character cryptographic secret | `openssl rand -hex 32` |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:////app/storage/parallax.db` |
| `STORAGE_DIR` | Absolute path to media storage | `/app/storage` |
| `ACCESS_TTL_MIN` | JWT access token lifetime | `15` |
| `REFRESH_TTL_DAYS` | Refresh token lifetime | `7` |
| `COOKIE_SECURE` | Enforce HTTPS-only cookies | `true` |
| `MAX_UPLOAD_MB` | Hard upload cap for video clips | `100` |
| `MAX_DURATION_S` | Maximum allowed video duration | `60` |
| `MODEL_PATH` | Path to model weights checkpoint | `ml/weights/avt2_dwf.pt` |
| `FEED_CACHE_TTL_S` | Feed cache expiration | `900` |
| `SAMPLE_HOST_ALLOWLIST` | Comma-separated host allowlist | `upload.wikimedia.org,archive.org` |
| `MAX_CONCURRENT_JOBS`| Worker concurrency limit | `2` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## 3. Container Deployment (Docker Compose)

The multi-stage `Dockerfile` compiles the production Vite frontend bundle and injects it into a hardened, non-root Python runtime image.

### Build and Launch:

```bash
# Build the container image and start in background
docker compose up -d --build

# Inspect running container status
docker compose ps

# View live container logs
docker compose logs -f parallax
```

The container automatically executes Alembic migrations (`alembic upgrade head`) before initiating Uvicorn workers.

---

## 4. Reverse Proxy & HTTPS Configuration (Nginx)

In production, place PARALLAX behind a hardened reverse proxy with TLS termination.

```nginx
server {
    listen 443 ssl http2;
    server_name parallax.example.com;

    ssl_certificate /etc/letsencrypt/live/parallax.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/parallax.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Client body limit matching MAX_UPLOAD_MB
    client_max_body_size 100M;

    # Security Headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none';" always;

    # Proxy to PARALLAX Container
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for Server-Sent Events (SSE)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## 5. Health Verification & Monitoring

### Liveness Probe
```bash
curl -f http://localhost:8000/api/v1/health
# Response: {"status":"healthy","timestamp":"..."}
```

### Readiness Probe
```bash
curl -f http://localhost:8000/api/v1/ready
# Response: {"ready":true,"db_connected":true,"model_loaded":true}
```

---

## 6. Rollback Procedures

If a deployment encounters issues:

1. **Revert Container Image**:
   ```bash
   docker compose down
   # Roll back to previous image tag or commit
   git checkout <PREVIOUS_COMMIT_TAG>
   docker compose up -d --build
   ```

2. **Database Migration Downgrade** (if required):
   ```bash
   docker compose exec parallax alembic -c backend/alembic.ini downgrade -1
   ```

3. **Storage Integrity Check**:
   Confirm that persistent volume `parallax_storage` retains file integrity:
   ```bash
   docker compose exec parallax ls -la /app/storage/
   ```
