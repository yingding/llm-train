# Reflections

## PyTorch XPU install strategy
- confidence: high
- source: observation
- scope: universal
- verified_count: 1
- last_verified: 2026-04-20
- derived_from: "Split pip requirements for mixed index sources" (conventions.md), "Intel XPU support in PyTorch" (project-setup.md), "Used --extra-index-url instead of split files initially" (anti-patterns.md)

**Context:** torch 2.11.0+ has built-in Intel XPU support; `intel-extension-for-pytorch` is deprecated.

**Pattern:** Split requirements into two files to isolate the PyTorch XPU index:
1. `requirements_winx64_pip_torch.txt` — uses `--index-url https://download.pytorch.org/whl/xpu` for torch, torchvision, torchaudio
2. `requirements_winx64_pip.txt` — uses `--index-url https://pypi.org/simple` for everything else
3. Install torch file first, then main file

**Why not a single file:** pip's `--extra-index-url` is global — it cannot restrict specific packages to a specific index. Packages may resolve from either index unpredictably.

**Anti-pattern:** Never use `--extra-index-url` when you need index isolation. Always split files with dedicated `--index-url` per file.
