# `prompt_evaluation.py`

## Purpose

Checks whether unseen prompt phrasings are semantically separable before expensive jewel fitting or
conditional training. It treats each class's training-template centroid as the only available label
prototype and reports held-out retrieval margins.

## Components

### `ClassPromptMetric` / `PromptGeometryReport`

- **Does**: records per-class unseen-template retrieval and aggregate accuracy/margins

### `evaluate_prompt_geometry`

- **Does**: validates the manifest digest, forms unit class centroids from training phrasings, and
  classifies every unseen evaluation phrase by cosine similarity
- **Rationale**: a failed text geometry gate cannot be repaired by fitting more jewel videos

### `main`

- **Does**: loads the manifest/cache and writes an auditable JSON report

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt smoke gate | Every held-out phrase retrieves its intended class with positive margin | Metric definition |
| Model evaluation | Training and evaluation phrasings remain separate | Centroid construction |
| Provenance | Embedding-cache digest must match the supplied manifest exactly | Digest check |
