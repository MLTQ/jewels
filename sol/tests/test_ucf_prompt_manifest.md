# `test_ucf_prompt_manifest.py`

## Purpose

Protects class/group parsing, longest-eligible selection, leakage-safe splitting, prompt separation,
and safe staging for the UCF prompt smoke corpus.

## Components

### `UCFPromptManifestTests`

- **Does**: verifies canonical IDs, balanced train/validation ownership, missing-group failure, and
  idempotent non-overwriting symlink behavior
- **Interacts with**: `ucf_prompt_manifest.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Smoke manifest | Four groups per class and longest eligible clip selection | Selection policy |
| Held-out evaluation | Group 4 appears only in validation and prompt templates do not overlap | Split fields |
| Remote fitter | Staging never replaces a regular file or redirects an existing symlink | Filesystem safety |
