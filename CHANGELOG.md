# Changelog

All notable changes to `llama-suite` should be documented in this file.

The milestone entries below now have matching local git tags. Push the tags and publish GitHub releases to make them visible on the repository page.

## Unreleased

### Added

- Added a reviewer-facing README with explicit `Why / How / What's Different` framing.
- Added real Web UI screenshots and an architecture diagram.
- Added a GitHub Actions CI workflow with a green-by-default test lane and advisory lint/type jobs.
- Added release documentation so the repo can be published with clearer portfolio signals.

## v0.3.0 - 2026-04-10

### Added

- Added Config Studio, parameter sweeps, GitOps mode, and deployment scaffolding for local and marketplace-style setups.
- Added richer Open WebUI controls, stop buttons, progress reporting, and improved results views.
- Added `ik_llama.cpp` runtime support and newer model/config coverage including Gemma 4 and Qwen 3.5 variants.

### Changed

- Hardened Linux setup and Open WebUI runtime fallback behavior.
- Simplified model config defaults and improved override-driven runtime handling.

## v0.2.0 - 2025-12-11

### Added

- Added the first FastAPI Web UI integration for local endpoint management.
- Introduced the `src/llama_suite/` package layout and started consolidating runtime helpers into the package.

### Changed

- Improved the updater flow to rebuild stale or broken virtual environments and handle macOS runtime library placement more reliably.

## v0.1.0 - 2025-08-11

### Added

- Established the Python-based watcher/runtime workflow for `llama-swap`.
- Added cross-platform config processing, benchmark utilities, and the initial local model operations foundation.

### Changed

- Reworked the repository structure and cleaned up earlier experimental scripts into a more maintainable layout.
