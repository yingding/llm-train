---
description: Episodic mistake buffer — specific incidents with correction links
priority: high
tags: [mistakes, corrections]
entry_count: 1
last_updated: 2026-04-20
---

# Anti Patterns

## Used --extra-index-url instead of split files initially
- confidence: high
- source: correction
- date: 2026-04-20
- supersedes: null
- corrected_by: "conventions.md#Split pip requirements for mixed index sources"

What happened: Initially put torch packages in the same requirements file with `--extra-index-url https://download.pytorch.org/whl/xpu`, which doesn't guarantee torch is installed exclusively from the XPU index.

Why it was wrong: pip's `--extra-index-url` is global and doesn't restrict which packages come from which index. Packages could resolve from either index unpredictably.

Lesson: When packages must come from a specific index, split into separate requirements files with their own `--index-url`. Never rely on `--extra-index-url` for index isolation.
