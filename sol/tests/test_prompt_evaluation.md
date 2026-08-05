# `test_prompt_evaluation.py`

## Purpose

Protects held-out prompt classification and manifest/cache provenance checks.

## Components

### `PromptEvaluationTests`

- **Does**: verifies perfect retrieval for separated synthetic class geometry and rejects caches
  built from a different manifest
- **Interacts with**: `prompt_evaluation.py` and `prompt_embeddings.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Text preflight | Positive held-out class-centroid margin indicates usable template geometry | Similarity protocol |
