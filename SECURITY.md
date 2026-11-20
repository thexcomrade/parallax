# PARALLAX — Security Architecture & Policy

This document outlines the security architecture, threat model mitigations, authentication invariants, and vulnerability reporting procedures for PARALLAX.

---

## 1. Authentication & Session Security

- **Password Hashing**: Uses **Argon2id** (`time_cost=3`, `memory_cost=65536`, `parallelism=4`, salt length 16 bytes). Plaintext passwords are never logged or stored.
- **Timing Attack Mitigation**: When authenticating unknown emails, a constant-time dummy Argon2 hash computation is performed to prevent account enumeration via timing differences.
- **Access Tokens**: Short-lived (15-minute) asymmetric/HMAC JWTs stored solely in client memory. Never stored in `localStorage` or `sessionStorage`.
- **Refresh Tokens & Family Revocation**: Stored in HTTP-only, `SameSite=Strict`, `Secure` cookies. Hashed using SHA-256 before database storage. Replay of an invalidated refresh token triggers immediate invalidation of the entire token family (forced session termination across all devices).

---

## 2. Multi-Tenant Data Isolation

- **Mandatory User Scoping**: Every database query retrieving analyses or forensic artifacts strictly filters by `user_id == current_user.id`.
- **Uniform 404 Responses**: Attempting to read, stream, or delete another user's analysis record returns a generic `404 Not Found` rather than `403 Forbidden`, preventing cross-tenant ID enumeration.

---

## 3. Media Ingestion & File Security

- **Multi-Layer File Validation**:
  1. File size cap enforcement (`MAX_UPLOAD_MB`).
  2. Magic byte / MIME signature inspection using `libmagic` byte matching (rejecting spoofed file extensions).
  3. `FFprobe` header decode validation executing strictly in a non-shell, bounded child process with execution timeouts.
- **Path Traversal Protection**: Uploaded files are stored using server-generated UUIDv4 filenames (`/storage/uploads/<uuid>.<ext>`). User-supplied filenames are never used in filesystem paths.
- **Automatic Failure Cleanup**: Files that fail validation or inference are deleted from storage immediately.

---

## 4. SSRF & External Media Staging Guard

When staging public domain reference clips (`/api/v1/samples/{id}/analyze`):
- **Host Allowlist**: Requests are restricted to predefined trusted domains (`upload.wikimedia.org`, `archive.org`).
- **IP Range Filtering**: Outbound requests resolving to loopback (`127.0.0.0/8`, `::1`), link-local (`169.254.0.0/16`), RFC 1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), or cloud metadata endpoints (`169.254.169.254`) are aborted immediately.
- **Redirect Guards**: HTTP redirects are inspected; foreign redirects outside the allowlist are denied.

---

## 5. Reporting a Security Vulnerability

If you discover a security vulnerability in PARALLAX:
1. Do not open a public GitHub issue.
2. Email security findings to `security@parallax.io` (or the project maintainer).
3. Include detailed steps to reproduce, affected versions, and proof-of-concept payloads.
4. Maintain responsible disclosure until an approved security patch is released.
