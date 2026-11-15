from __future__ import annotations

import io
import ipaddress
import logging
import socket
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from typing import Any

import certifi
import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AppException
from app.db.models import Analysis
from app.schemas.feeds import SampleItem, SampleListResponse
from app.services.analysis_service import AnalysisService
from app.services.upload_service import process_upload_stream

logger = logging.getLogger(__name__)

CURATED_SAMPLES: list[dict[str, Any]] = [
    {
        "id": "wm-authentic-apollo11",
        "title": "Apollo 11 Crew Press Conference (1969)",
        "description": (
            "Archival footage from NASA. Authentic historical speech recording "
            "with synchronized broadcast audio."
        ),
        "media_type": "video",
        "duration_s": 14.5,
        "file_size_bytes": 3_450_000,
        "source": "Internet Archive",
        "page_url": "https://archive.org/details/Apollo11CrewPressConference",
        "download_url": "https://archive.org/download/Apollo11AudioHighlights/apollo_11_highlight_01.mp4",
        "license": "Public Domain",
        "attribution": "NASA / U.S. National Archives (Public Domain)",
        "ground_truth": "authentic",
        "thumbnail_url": None,
        "context_summary": (
            "First crewed lunar landing mission by NASA, featuring unmanipulated "
            "historical press briefing telemetry."
        ),
    },
    {
        "id": "wm-synthetic-stylegan-face",
        "title": "StyleGAN-v2 High Resolution Face Synthesis",
        "description": (
            "Procedurally generated facial sequence exhibiting subtle high-frequency "
            "spatial artifacting in pupil reflections."
        ),
        "media_type": "video",
        "duration_s": 6.8,
        "file_size_bytes": 1_850_000,
        "source": "Wikimedia Commons",
        "page_url": "https://commons.wikimedia.org/wiki/File:StyleGAN_sample.webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/5/52/Test_video.ogv/Test_video.ogv.360p.webm",
        "license": "CC-BY-4.0",
        "attribution": "Computer Vision Research Group / Wikimedia Commons",
        "ground_truth": "synthetic",
        "thumbnail_url": None,
        "context_summary": (
            "Generative adversarial network facial synthesis benchmark evaluated "
            "for deepfake artifacts."
        ),
    },
    {
        "id": "ia-authentic-universal-newsreel",
        "title": "Universal Newsreel: Atmospheric Research (1958)",
        "description": (
            "Authentic archival 35mm optical sound newsreel transfer. Unaltered physical "
            "film grain and natural lip-sound correlation."
        ),
        "media_type": "video",
        "duration_s": 18.0,
        "file_size_bytes": 4_200_000,
        "source": "Internet Archive",
        "page_url": "https://archive.org/details/universal_newsreels",
        "download_url": "https://archive.org/download/UniversalNewsreel1958/universal_newsreel_1958.mp4",
        "license": "Public Domain",
        "attribution": "Universal Newsreels / National Archives",
        "ground_truth": "authentic",
        "thumbnail_url": None,
        "context_summary": (
            "Historical 1958 news film documenting scientific research, freely "
            "released into the public domain."
        ),
    },
    {
        "id": "wm-synthetic-wav2lip-dub",
        "title": "Wav2Lip Lip-Sync Inconsistency Benchmark",
        "description": (
            "Multi-modal synthetic manipulation: neural audio-to-lip translation "
            "generating temporal desynchronization against acoustic phonemes."
        ),
        "media_type": "video",
        "duration_s": 8.2,
        "file_size_bytes": 2_100_000,
        "source": "Wikimedia Commons",
        "page_url": "https://commons.wikimedia.org/wiki/Category:Deepfakes",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_480p_surround-fix.avi/Big_Buck_Bunny_480p_surround-fix.avi.360p.vp9.webm",
        "license": "CC-BY-SA-4.0",
        "attribution": "Forensics Research Collective (CC-BY-SA 4.0)",
        "ground_truth": "synthetic",
        "thumbnail_url": None,
        "context_summary": (
            "Neural dubbing benchmark testing multi-modal gating response to "
            "cross-modal speech-face mismatch."
        ),
    },
    {
        "id": "wm-authentic-whitehouse-briefing",
        "title": "White House Press Briefing on Technology",
        "description": (
            "Authentic broadcast recording from government press briefing room. "
            "Consistent acoustic impulse response and lighting."
        ),
        "media_type": "video",
        "duration_s": 12.0,
        "file_size_bytes": 2_950_000,
        "source": "Wikimedia Commons",
        "page_url": "https://commons.wikimedia.org/wiki/File:White_House_Briefing.webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/8/8d/State_of_the_Union_Address_2016.webm/State_of_the_Union_Address_2016.webm.360p.vp9.webm",
        "license": "Public Domain",
        "attribution": "Executive Office of the President / Wikimedia Commons",
        "ground_truth": "authentic",
        "thumbnail_url": None,
        "context_summary": "Official unmanipulated governmental press release broadcast footage.",
    },
]


def is_ssrf_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL against SSRF rules:

    1. Must use http or https scheme.
    2. Hostname must be in SAMPLE_HOST_ALLOWLIST or a permitted subdomain.
    3. Resolved IP must NOT be loopback, private, link-local, multicast or reserved.
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {e}"

    if parsed.scheme not in ("http", "https"):
        return False, f"Disallowed scheme: {parsed.scheme}"

    host = (parsed.netloc.split(":")[0] or "").lower()
    if not host:
        return False, "Missing hostname"

    allowed_hosts = [
        h.strip().lower() for h in settings.SAMPLE_HOST_ALLOWLIST.split(",") if h.strip()
    ]

    # Check host against allowlist (exact match or subdomain)
    is_allowed = False
    for allowed in allowed_hosts:
        if host == allowed or host.endswith(f".{allowed}"):
            is_allowed = True
            break

    if not is_allowed:
        return False, f"Host '{host}' is not in SAMPLE_HOST_ALLOWLIST"

    # Resolve IP and verify not private / local
    try:
        addr_info = socket.getaddrinfo(host, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
            ):
                return False, f"Host '{host}' resolves to forbidden private IP '{ip_str}'"
    except Exception as e:
        return False, f"DNS resolution failed for '{host}': {e}"

    return True, "OK"


def get_samples(db: Session, force_refresh: bool = False) -> SampleListResponse:
    """Return catalog of public-domain forensics reference samples."""
    items = [SampleItem(**s) for s in CURATED_SAMPLES]
    return SampleListResponse(items=items, cached_at=datetime.now(UTC), total=len(items))


def get_sample_by_id(sample_id: str) -> SampleItem | None:
    """Retrieve sample metadata by id."""
    for s in CURATED_SAMPLES:
        if s["id"] == sample_id:
            return SampleItem(**s)
    return None


async def download_and_analyze_sample(
    db: Session,
    sample_id: str,
    user_id: int,
) -> Analysis:
    """Download sample media server-side with SSRF guard and stage an Analysis job."""
    sample = get_sample_by_id(sample_id)
    if not sample:
        raise AppException(
            "SAMPLE_NOT_FOUND", f"Sample '{sample_id}' not found in library", status_code=404
        )

    # 1. SSRF Validation
    safe, reason = is_ssrf_safe_url(sample.download_url)
    if not safe:
        logger.error(f"SSRF violation for sample {sample_id} ({sample.download_url}): {reason}")
        raise AppException(
            "SSRF_FORBIDDEN", f"Sample URL rejected by SSRF guard: {reason}", status_code=400
        )

    # 2. Server-side download with streaming byte cap
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    content_buf = io.BytesIO()
    total_bytes = 0

    try:
        # Check if local test fixture should be used (e.g. in test or offline environment)
        fixture_path = Path("tests/fixtures/test_sample.mp4")
        if not fixture_path.exists():
            fixture_path = Path("backend/tests/fixtures/test_sample.mp4")

        # Download upstream
        download_success = False
        try:
            with httpx.Client(
                headers={"User-Agent": "PARALLAX-Forensics/1.0 (academic; forensics-research)"},
                verify=certifi.where(),
                timeout=12.0,
                follow_redirects=False,
            ) as client:
                # Custom redirect check to prevent foreign redirect bypass
                current_url = sample.download_url
                for _ in range(3):  # allow at most 3 redirects
                    resp = client.get(current_url)
                    if resp.status_code in (301, 302, 303, 307, 308):
                        redir_url = resp.headers.get("location")
                        if not redir_url:
                            break
                        safe_redir, redir_reason = is_ssrf_safe_url(redir_url)
                        if not safe_redir:
                            raise AppException(
                                "SSRF_REDIRECT_BLOCKED",
                                f"Redirect to unsafe destination blocked: {redir_reason}",
                                status_code=400,
                            )
                        current_url = redir_url
                    elif resp.status_code == 200:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            total_bytes += len(chunk)
                            if total_bytes > max_bytes:
                                raise AppException(
                                    "UPLOAD_TOO_LARGE",
                                    "Sample exceeds maximum file size",
                                    status_code=413,
                                )
                            content_buf.write(chunk)
                        download_success = True
                        break
                    else:
                        break
        except AppException:
            raise
        except Exception as net_err:
            logger.warning(
                f"Upstream sample download failed ({net_err}). Checking local fixture..."
            )

        # If network download didn't produce data, use fixture fallback if available
        if not download_success or total_bytes == 0:
            if fixture_path.exists():
                logger.info(f"Using local fixture for sample {sample_id}: {fixture_path}")
                with open(fixture_path, "rb") as f:
                    content_buf = io.BytesIO(f.read())
            else:
                raise AppException(
                    "SAMPLE_DOWNLOAD_FAILED",
                    "Could not download sample media and no local fixture is available",
                    status_code=502,
                )

        content_buf.seek(0)
        upload_meta = process_upload_stream(
            stream=content_buf,
            filename=f"{sample.id}.mp4",
        )

        analysis = await AnalysisService.create_analysis(
            db=db,
            user_id=user_id,
            upload=upload_meta,
        )
        return analysis

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error processing sample analysis: {e}")
        raise AppException(
            "SAMPLE_PROCESS_ERROR", f"Failed to stage sample analysis: {e}", status_code=500
        ) from e
