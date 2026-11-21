# Changelog

All notable changes to the PARALLAX Audio-Visual Deepfake Forensics system are documented in this file.

---

## [1.2.0] — 2026-11-21

### Frontend A–Z Audit & Design-System Consolidation
- **Single token layer**: `frontend/src/styles/tokens.css` is now the only colour/spacing/motion source. Removed the second ad-hoc palette (611 hard-coded hex values across 13 files) and the duplicate `@theme`/`:root` declarations; added `--space-*`, `--success-*`, `--warning-*`, `--accent-*` and the `xs` breakpoint.
- **Design decisions**: primary action is Cobalt everywhere (previously lime on auth screens, blue elsewhere); `AUTHENTIC` is Emerald; amber reserved for untrained/inconclusive. `docs/design/DESIGN.md` §2/§7/§8 updated to match; lime retired.
- **Authentication**: rebuilt `/login` and `/register` on the shared `Field`/`Button` primitives with the real modality specification, removed the “Welcome back” copy, decorative blurred blobs, empty badge and unverifiable “256-bit rotating sessions” claim.
- **Contract & security**: analysis JSON export and history CSV no longer include `p_fake`; report deletion moved from `window.confirm`/`alert` to the focus-trapped `Dialog`.
- **Dead code**: deleted the unused `components/viz` and `components/bento` trees, `useCursorSync`, `useAuthenticatedImage`, `Tabs`, the unused UI barrel and `PageHeader` (≈1,650 lines of source). None of them was reachable from `routes.tsx`, so the shipped bundle is unchanged; they also rendered internal `p_fake`, modality weights and sync scores, which is why removal matters for the contract.
- **Accessibility/semantics**: drop surface is a single native button (no clickable `div`), dashboard ingestion tile is a link, mobile nav drawer/backdrop and popovers got proper dialog/tooltip semantics, focus ring + reduced-motion verified.
- **Docs**: `experiments/phase7_deployment/FRONTEND_COMPLETE_CODE_AUDIT.md` and `FRONTEND_REDESIGN_REPORT.md`.

---

## [1.1.0] — 2026-11-21

### Phase 7 & Post-Deployment Remediation: Production Hardening, Branding & Infrastructure Architecture
- **Brand Identity & UI System**: Redesigned official PARALLAX vector mark (`ParallaxLogo.tsx`) featuring precision dual-stream optical prism and acoustic harmonic facets with central dynamic weight fusion core. Replaced generic icons across CommandRail, StatusStrip, LoginPage, and RegisterPage, alongside crisp 32x32 SVG favicon.
- **Authentication UX Remediation**: Resolved onboarding and login credential ambiguity by introducing explicit analyst registration routing, instructional hints for new operators, and targeted inline guidance for failed credentials.
- **Repository Architecture & Hygiene**: Overhauled root `.gitignore` to protect production weights and evaluation manifests while excluding build artifacts, local secrets, and caches. Cleared stale `.pyc` files and empty temporary directories.
- **Enterprise Deployment & Orchestration**: Containerized staging deployment using PostgreSQL 16, Redis 7 message queue, asynchronous distributed GPU inference worker, Nginx reverse proxy, and Cloudflare Quick Tunnel orchestration.
- **Disaster Recovery & Backup Consolidation**: Centralized disaster recovery assets into `D:\Projects\parallax-backup\` with cryptographic SHA-256 verification and automated restoration validation.

---

## [1.0.0] — 2026-11-16

### Phase 15: Final Production Audit & Release Readiness
- Audited complete full-stack repository architecture, dependencies, security invariants, and test coverage.
- Validated all 63 Python backend tests (Pytest) and 55 Playwright E2E frontend tests across 5 viewports with 100% pass rate.
- Verified zero `console.log` statements, zero `any` types, zero `@ts-ignore` comments, and zero `print` calls in application source code.
- Authored production operations runbooks: `DEPLOYMENT.md` and `SECURITY.md`.

### Phase 14: Advanced Responsive Engineering & Motion System
- Implemented standardized micro-interaction motion tokens (`--motion-fast`, `--motion-normal`, `--motion-smooth`, `--ease-standard`, `--ease-emphasized`) in `tokens.css`.
- Configured `.interactive-card` and `.interactive-btn` elevation utilities with complete `prefers-reduced-motion` compliance.
- Resolved mobile popover alignment in `StatusStrip.tsx` to eliminate viewport clipping on 360px and 390px screens.
- Completely removed obsolete Model Card references, navigation items, routes, and feature components per operator request.

### Phase 13: Settings Page Redesign
- Redesigned Operator Settings into a modular layout covering profile information, session controls, password updates, and accessible purge dialogs with focus trapping.

### Phase 12: Model Audit & Cleanup
- Audited model documentation endpoints and removed unneeded model specifications from UI navigation.

### Phase 11: Library & Reference Corpus Refinement
- Refined public domain reference corpus workbench with classification filters, responsive metadata cards, and direct sample staging.

### Phase 10: Training & Evaluation Architecture
- Implemented offline dataset splitting, multimodal augmentation, balanced batch sampling, and evaluation pipelines for AVT²-DWF.

### Phase 9: Dashboard & Telemetry Instrumentation
- Implemented responsive bento grid dashboard with activity histograms, runtime diagnostics, and recent detection ledgers.

### Phase 8: Public Threat Intel Feeds
- Integrated real-time threat radar, arXiv forensic research papers, and technology discussion feeds with memory caching.

### Phase 7: Analyze & Report Forensics
- Built drag-and-drop ingestion workbench with viewfinder guidelines and multi-modal forensic inspection report.

### Phase 6: Analysis Pipeline & Job Runner
- Built bounded async worker queue with Server-Sent Events (SSE) streaming real-time stage progress to client.

### Phase 5: Model Inference & Explainability
- Assembled Face Transformer Encoder, Audio Transformer Encoder, and Dynamic Weight Fusion (DWF) into AVT²-DWF with attention rollout heatmaps.

### Phase 4: Media Preprocessing
- Built FFprobe validation, face detection/tracking, and audio MFCC/log-mel spectrogram extraction modules.

### Phase 3: Authentication & Protected Shell
- Implemented Argon2id credential hashing, rotating refresh tokens with token-family replay revocation, and responsive application shell.

### Phase 2: Core Backend & Database
- Configured FastAPI application, SQLite WAL database engine, SQLAlchemy 2 ORM tables, and Alembic migrations.

### Phase 1: Design System
- Established zero-template dark design system tokens in `DESIGN.md` and `tokens.css`.

### Phase 0: Repository & Tooling
- Initialized repository architecture, TypeScript strict configuration, Ruff linter, and Playwright E2E framework.
