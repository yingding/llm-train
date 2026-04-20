---
description: Development process rules and decision-making
priority: high
tags: [process, decisions]
entry_count: 1
last_updated: 2026-04-20
---

# Workflow

## Environment setup documentation structure
- confidence: high
- source: observation
- verified_count: 1
- last_verified: 2026-04-20

Env setup docs split by package manager:
- `envtools/README_winconda.md` — conda-based setup with `conda env update --prefix`
- `envtools/README_winpip.md` — pip-based setup
- `envtools/create_env.ps1` / `create_env.sh` — shared env creation scripts
- Conda envs use `environment_winx64*.yml`, pip envs use `requirements_winx64_pip*.txt`
