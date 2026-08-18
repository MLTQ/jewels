# `cache_encoder_latents.py`

## Purpose

Stage 1's data step: freezes a trained amortized encoder and precomputes the generation target
— one latent per corpus window — alongside token-level prompt embeddings. Training a
text-conditioned generator then needs no video decoding and no encoder forward pass.

## Components

### `encode_prompts`
- **Does**: Encodes every unique prompt with a small HF text model, returning padded
  **token sequences** plus attention masks (not a pooled vector).
- **Rationale**: Pooled sentence embeddings are precisely what the earlier CLIP-conditioned
  attempts failed with — verbs and composition survive tokenization but not pooling. Cross
  attention over tokens is the architecture the roadmap selected.

### `main`
- **Does**: Loads the encoder checkpoint, `encode()`s every manifest window, and saves a single
  `latents.pt` holding stacked `cells`/`seed` tensors, prompt token embeddings, per-window
  records (style, class, split, prompt index), and full encoder/text-model provenance.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Stage 1 generator trainer | Schema `jewel-encoder-latent-cache-v1`; stacked latents aligned with `records` | Field names |
| Decode path | `cells`/`seed` shapes match the recorded `model_args` and grid | Encoder change |

## Notes

- Defaults to `BAAI/bge-small-en-v1.5`, already present in the host cache, so no download is
  needed on a disk-constrained box.
- Distinct from the retired-era `cache_latents.py` (tokenizer lane); this one caches
  amortized-encoder latents.
