"""PARALLAX — Centralized Structured JSON Logging & Observability.

Provides high-performance, structured JSON logging with:
- UTC ISO 8601 timestamps
- Contextual request ID (`X-Request-ID`) and analysis/job ID correlation
- Process role (`PROCESS_ROLE`) and PID attribution
- Error stack traces formatted into structured fields
- Automatic redaction of sensitive credentials, tokens, passwords, and signed URLs
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

from app.core.config import settings

# Context variables for asynchronous request/job correlation
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
job_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "job_id", default=None
)

# Regex patterns for sensitive data redaction
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|token|secret|authorization|signature|credential|api[_-]?key|access[_-]?key)",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"Bearer\s+([A-Za-z0-9._~+/-]+=*)", re.IGNORECASE)
_PRESIGNED_S3_PATTERN = re.compile(
    r"((?:X-Amz-Signature|Signature)=[A-Za-z0-9%_-]+)",
    re.IGNORECASE,
)
_CREDENTIAL_PARAM_PATTERN = re.compile(
    r"((?:X-Amz-Credential|access_token|refresh_token)=[A-Za-z0-9%_/-]+)",
    re.IGNORECASE,
)


def redact_sensitive_string(text: str) -> str:
    """Sanitize sensitive substrings from log messages."""
    if not text:
        return text
    text = _BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    text = _PRESIGNED_S3_PATTERN.sub("Signature=[REDACTED]", text)
    text = _CREDENTIAL_PARAM_PATTERN.sub(r"\1=[REDACTED]", text)
    return text


def sanitize_value(key: str, value: Any) -> Any:
    """Recursively redact sensitive keys and values from logging dictionaries."""
    if _SENSITIVE_KEY_PATTERN.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_sensitive_string(value)
    if isinstance(value, dict):
        return {str(k): sanitize_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize_value(key, item) for item in value]
    return value


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as structured, single-line JSON objects."""

    def __init__(self) -> None:
        super().__init__()
        self._pid = os.getpid()

    def format(self, record: logging.LogRecord) -> str:
        # 1. Base log record fields
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        raw_message = record.getMessage()
        sanitized_message = redact_sensitive_string(raw_message)

        log_obj: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_message,
            "process_id": self._pid,
            "process_role": settings.PROCESS_ROLE,
        }

        # 2. Contextual Request ID and Job ID correlation
        req_id = getattr(record, "request_id", None) or request_id_ctx.get()
        if req_id:
            log_obj["request_id"] = req_id

        analysis_id = (
            getattr(record, "analysis_id", None)
            or getattr(record, "job_id", None)
            or job_id_ctx.get()
        )
        if analysis_id:
            log_obj["analysis_id"] = analysis_id

        worker_id = getattr(record, "worker_id", None)
        if worker_id:
            log_obj["worker_id"] = worker_id

        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms is not None:
            log_obj["duration_ms"] = round(float(duration_ms), 2)

        # 3. Source location for warnings and errors
        if record.levelno >= logging.WARNING:
            log_obj["source"] = f"{record.filename}:{record.lineno} ({record.funcName})"

        # 4. Exception stack trace handling
        if record.exc_info:
            exc_type, exc_val, exc_tb = record.exc_info
            if exc_val is not None:
                log_obj["exception"] = {
                    "type": getattr(exc_type, "__name__", "Exception"),
                    "message": redact_sensitive_string(str(exc_val)),
                    "stacktrace": [
                        redact_sensitive_string(line)
                        for line in traceback.format_exception(exc_type, exc_val, exc_tb)
                    ],
                }

        # 5. Extract extra custom fields
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "request_id", "analysis_id",
            "job_id", "worker_id", "duration_ms",
        }
        custom_extra: dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                custom_extra[key] = sanitize_value(key, val)

        if custom_extra:
            log_obj["extra"] = custom_extra

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(log_level: str | None = None) -> None:
    """Initialize structured logging across the root and application loggers."""
    level_name = (log_level or settings.LOG_LEVEL).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    # Configure stdout stream handler with StructuredJsonFormatter
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(StructuredJsonFormatter())
    stream_handler.setLevel(numeric_level)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    # Remove any existing standard handlers to avoid duplicate text output
    root_logger.handlers = [stream_handler]

    # Specific namespace loggers
    for namespace in ("parallax", "app", "uvicorn.access", "uvicorn.error"):
        sub_logger = logging.getLogger(namespace)
        sub_logger.setLevel(numeric_level)
        sub_logger.handlers = []
        sub_logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """Convenience getter for namespaced Parallax loggers."""
    if not name.startswith("parallax"):
        name = f"parallax.{name}"
    return logging.getLogger(name)


def set_request_context(request_id: str | None) -> None:
    """Set the asynchronous request ID context for the active coroutine."""
    request_id_ctx.set(request_id)


def set_job_context(job_id: str | None) -> None:
    """Set the asynchronous analysis/job ID context for the active coroutine."""
    job_id_ctx.set(job_id)
