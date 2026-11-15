# Multi-stage production container for PARALLAX (README §8.1, §13)

# ------------------------------------------------------------------------------
# Stage 1: Build production frontend bundle
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python runtime + CPU PyTorch + FFmpeg + non-root user
# ------------------------------------------------------------------------------
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    STORAGE_DIR=/app/storage

# Install system dependencies: ffmpeg, media libraries, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root application user
RUN groupadd -g 10001 parallax && \
    useradd -u 10001 -g parallax -s /bin/bash -m parallax

WORKDIR /app

# Install CPU PyTorch first to keep layer cached and image slim
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY backend/pyproject.toml ./backend/
RUN pip install --no-cache-dir -e "./backend"

# Copy backend source code & migrations
COPY backend/ ./backend/

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Setup persistent storage directory with correct ownership
RUN mkdir -p /app/storage/uploads /app/storage/artifacts && \
    chown -R parallax:parallax /app

# Switch to non-root user
USER parallax

# Container exposes HTTP on port 8000
EXPOSE 8000

# Health check against production endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Apply DB migrations then start Uvicorn
CMD ["sh", "-c", "alembic -c backend/alembic.ini upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend"]
