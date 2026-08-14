# `rebind_prompt_cache.py`

## Purpose

Attaches an already-encoded prompt cache to a derived manifest (e.g. a scaling subset) whose
prompt text is unchanged. This keeps embeddings bit-identical across every curve point — better
controlled than re-encoding — and needs no text-encoder installation.

## Components

### `main`
- **Does**: Verifies the parent cache owns the parent manifest, then rebuilds ownership indices
  and the digest for the derived manifest via `build_prompt_cache`, which itself refuses any
  manifest whose collected prompt tuple differs from the parent's.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scaling-curve trainers | Subset caches share the parent's embedding rows exactly | Re-encoding drift |
| `prompt_embeddings.py` | `collect_prompts(subset) == parent.prompts` or the rebind fails | Prompt-order rule |
