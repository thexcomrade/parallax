# PARALLAX Load Testing Suite (k6)

This directory contains the k6 load testing suite for the PARALLAX Multimodal Audio-Visual Deepfake Detection System

## Structure
- `config/`: Target and threshold configurations (`staging.json`, `thresholds.json`).
- `scenarios/`: Individual scenario tests:
  - `health.js`: Liveness and readiness endpoints.
  - `authentication.js`: User login, token rotation, and profile queries.
  - `metadata.js`: System metadata and model card retrieval.
  - `upload.js`: Multipart video file uploads.
  - `job_submission.js`: Job enqueueing and status polling.
  - `inference.js`: Full end-to-end pipeline execution and public binary verdict validation.
  - `mixed_workload.js`: Realistic multi-path workload distribution.
- `lib/`: Shared utilities for authentication, client security validation, fixtures, checks, and metrics.
- `stages/`: Progressive stage definitions from 50 to 30,000 VUs.
- `distributed/`: Configuration templates and coordinator script for multi-node distributed k6 generation.
- `fixtures/`: Disposable synthetic media files (never uses research datasets).

## Safety Rules
1. Never direct load tests against production endpoints (`BASE_URL` allowlist strictly enforced).
2. Never modify model weights (`backend/app/ml/weights/avt2_dwf.pt`).
3. Never access locked test manifests.
4. Single-host generator execution is capped at sustainable limits; 5K–30K stages require external generator nodes.
