# Rules for every task in this repo

## Process
1. Read this README fully before the first edit. Work one phase at a time; do not start the next phase.
2. Before creating a file, check it is listed in README §8. If you need a file that is not listed, say why first.
3. Create only files that are needed now. No placeholder files, no empty "future" modules, no sample junk, no duplicate helpers.
4. After each phase: run its "Done when" checks, fix failures, and summarise in 5 lines. Do not suppress errors to pass checks.
5. Commit-sized changes. Never touch unrelated files.

## Code
- Backend: Python 3.12, FastAPI, SQLAlchemy 2, Pydantic v2, full type hints, `ruff` + `mypy --strict` clean.
- Frontend: React 18 + TypeScript strict + Vite + Tailwind v4 + Motion. No `any`, no console logs, no commented-out code.
- Secrets only from environment variables. `.env` is git-ignored; only `.env.example` (names, no values) is committed.
- All third-party/public API calls go through the backend (`feed_service`, `sample_service`). The browser only talks to our own origin.
- Every query that returns user data filters by the authenticated user id. Never trust a user id sent by the client.
- Never fabricate model metrics, results, or sample data presented as real.

## Design (zero "AI template" look)
- Follow `docs/design/DESIGN.md` (created in Phase 1). Tokens only; no hard-coded colours/sizes in components.
- Forbidden: purple/pink gradients, glassmorphism cards, floating blurred blobs, emoji in UI, Inter/Roboto/Poppins/Space Grotesk as primary fonts, centered hero + three identical feature cards, stock "Welcome back 👋" copy, identical rounded-2xl cards everywhere, lorem ipsum, generic icon-in-circle feature rows.
- Motion: transform/opacity only, 120–400 ms, `cubic-bezier(.2,.8,.2,1)`, respect `prefers-reduced-motion`, pause canvases when the tab is hidden.
- Accessibility: WCAG AA contrast, visible focus ring, keyboard-complete flows, semantic landmarks, alt text/aria labels.
- Copy is short, specific and technical. Labels read like instrument panels ("CH-V", "SYNC 0.82"), not marketing.

## Done means
Lint, types, unit/e2e tests and a production build all pass, and the feature works in the browser at 390 / 768 / 1280 / 1920 px.
