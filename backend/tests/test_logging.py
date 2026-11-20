"""PARALLAX — Centralized Structured JSON Logging Tests."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.logging import (
    StructuredJsonFormatter,
    job_id_ctx,
    redact_sensitive_string,
    request_id_ctx,
    sanitize_value,
    set_job_context,
    set_request_context,
)


def test_structured_json_formatter_valid_json() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="parallax.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test log message with id %s",
        args=("12345",),
        exc_info=None,
    )

    formatted = formatter.format(record)
    parsed: dict[str, Any] = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "parallax.test"
    assert parsed["message"] == "Test log message with id 12345"
    assert "timestamp" in parsed
    assert "process_id" in parsed
    assert "process_role" in parsed


def test_structured_logging_context_propagation() -> None:
    formatter = StructuredJsonFormatter()

    set_request_context("req-abc-999")
    set_job_context("job-xyz-111")

    try:
        record = logging.LogRecord(
            name="parallax.api",
            level=logging.INFO,
            pathname="test.py",
            lineno=25,
            msg="Processing user analysis request",
            args=(),
            exc_info=None,
        )
        record.duration_ms = 45.2  # type: ignore[attr-defined]

        formatted = formatter.format(record)
        parsed: dict[str, Any] = json.loads(formatted)

        assert parsed["request_id"] == "req-abc-999"
        assert parsed["analysis_id"] == "job-xyz-111"
        assert parsed["duration_ms"] == 45.2
    finally:
        request_id_ctx.set(None)
        job_id_ctx.set(None)


def test_sensitive_information_redaction_strings() -> None:
    # 1. Bearer Token Redaction
    token_str = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID"
    redacted_token = redact_sensitive_string(token_str)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted_token
    assert "Bearer [REDACTED]" in redacted_token

    # 2. Pre-signed S3 Signature Redaction
    s3_url = "https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Signature=abcd1234ef567890"
    redacted_url = redact_sensitive_string(s3_url)
    assert "abcd1234ef567890" not in redacted_url
    assert "Signature=[REDACTED]" in redacted_url


def test_sensitive_dictionary_and_extra_sanitization() -> None:
    payload = {
        "user_email": "operator@parallax.internal",
        "password": "SuperSecretPassword123!",
        "access_token": "token_abc_xyz",
        "nested": {
            "secret_key": "raw_secret_here",
            "safe_field": "safe_value",
        },
    }

    sanitized = sanitize_value("data", payload)

    assert sanitized["user_email"] == "operator@parallax.internal"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["access_token"] == "[REDACTED]"
    assert sanitized["nested"]["secret_key"] == "[REDACTED]"
    assert sanitized["nested"]["safe_field"] == "safe_value"


def test_structured_exception_stacktrace_formatting() -> None:
    formatter = StructuredJsonFormatter()

    try:
        raise ValueError("Invalid pipeline dimension token: Bearer secret-1234")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="parallax.worker",
            level=logging.ERROR,
            pathname="worker.py",
            lineno=50,
            msg="Task crashed during processing",
            args=(),
            exc_info=sys.exc_info(),
        )

    formatted = formatter.format(record)
    parsed: dict[str, Any] = json.loads(formatted)

    assert parsed["level"] == "ERROR"
    assert "exception" in parsed
    assert parsed["exception"]["type"] == "ValueError"
    assert "secret-1234" not in parsed["exception"]["message"]
    assert "[REDACTED]" in parsed["exception"]["message"]
    assert isinstance(parsed["exception"]["stacktrace"], list)
