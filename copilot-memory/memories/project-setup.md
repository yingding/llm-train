---
description: Project-specific configuration and structure notes
priority: high
tags: [config, structure]
entry_count: 2
last_updated: 2026-04-20
---


# Project Setup

## Intel XPU support in PyTorch
- confidence: high
- source: conversation
- verified_count: 1
- last_verified: 2026-04-20

As of torch 2.11.0, Intel XPU support is built directly into PyTorch. `intel-extension-for-pytorch` is deprecated and should not be used. Install torch with `--index-url https://download.pytorch.org/whl/xpu` to get XPU-enabled builds.

## Current project versions and env layout
- confidence: high
- source: observation
- verified_count: 1
- last_verified: 2026-04-20

Key versions: torch==2.11.0, torchvision==0.26.0, torchaudio==2.11.0, Python 3.13. Working branch: `xpu`. Training venvs stored at `$USERPROFILE\Documents\VENV\`. Env naming: `{name}{version}{suffix}` e.g. `train3.13winconda`, `train3.13pip`.
