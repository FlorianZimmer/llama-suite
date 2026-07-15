# Python package scope

Inherit the repository-root `AGENTS.md`. This tree owns the installed Python
control plane: configuration, runtime discovery/processes, eval and benchmark
orchestration, proxy behavior, and the FastAPI Web UI.

- Keep side effects at explicit runtime boundaries so config parsing and unit
  tests remain deterministic and do not require models, servers, or network.
- Resolve paths and machine differences through the established root/config
  helpers; do not add workstation-specific constants to package code.
- Preserve API authentication and subprocess argument boundaries when changing
  Web UI, proxy, watcher, or command-building code.
- Test the affected module and its public consumer. Widen to the full suite only
  when shared config, registry, process, or Web UI infrastructure changes.
- Follow the root validation and hygiene rules; this file adds no separate
  formatting, planning, or tool ceremony.
