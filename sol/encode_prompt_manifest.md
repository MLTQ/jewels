# `encode_prompt_manifest.py`

## Purpose

Encodes all training and held-out prompt templates in a streaming manifest with its declared frozen
OpenCLIP text tower and writes a validated prompt sidecar.

## Components

### `main`

- **Does**: restores the exact declared encoder, batches text tokenization, unit-normalizes vectors,
  builds manifest ownership indices, and atomically saves the cache
- **Interacts with**: `prompt_embeddings.py` and `ucf_prompt_manifest.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt trainer | Cache embedding dimension becomes `BirthContinuationModel.text_dim` | Encoder/model change |
| Correct/shuffled controls | All compared prompts come from one frozen normalized space | Normalization |
| Reproducibility | Encoder name, pretrained weights, and manifest digest are serialized | Provenance fields |

## Notes

- OpenCLIP is imported only inside the CLI so tests and non-prompt workflows do not require it.
