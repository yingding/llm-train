---
description: Coding standards, patterns, and project conventions
priority: high
tags: [standards, patterns]
entry_count: 1
last_updated: 2026-04-20
---

# Conventions

## Split pip requirements for mixed index sources
- confidence: high
- source: conversation
- verified_count: 1
- last_verified: 2026-04-20

A single requirements.txt cannot restrict specific packages to specific indexes — `--index-url` and `--extra-index-url` are global. To ensure torch packages come only from the PyTorch XPU index, split into two files:
- `requirements_winx64_pip_torch.txt` with `--index-url https://download.pytorch.org/whl/xpu`
- `requirements_winx64_pip.txt` with `--index-url https://pypi.org/simple`
Install order: torch file first, then main file.
