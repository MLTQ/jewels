# label_corpus.py

## Purpose
Conditioning data for t2v: attach a CLIP image embedding to every fitted window. The stage-2
prior conditions on these (with dropout → classifier-free guidance; unconditional generation
is the dropped-out case). CLIP specifically because its text encoder shares the embedding
space — when a diverse captioned corpus exists, text conditioning drops into the SAME
interface with no prior retraining semantics change.

## Components

### `main()`
- **Does**: for each `*_w*.pt`, load a few source frames (via the checkpoint's `source` key),
  encode with open_clip, mean-pool, save `<ckpt>.clip.npy` sidecar. Resumable; skips existing.
- **Interacts with**: `cli/fit_corpus.py` checkpoints (needs their `source` traceability),
  `data/video_io.load_video(start_frame=...)`

## Decisions
- Sidecar `.npy` files, not embedded in the `.pt`: labeling reruns (better CLIP, more frames,
  a future video encoder) must not touch fitted checkpoints.
- Mean of a few frame embeddings — the standard cheap video embedding. Per-frame or temporal
  encoders are a later refinement; single-scene corpora wouldn't reward them yet.
- ViT-B-32 / laion2b default: small, fast, good enough for plumbing validation.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| stage 2 (future) | `<stem>_w<start>.clip.npy` float32 unit-norm | Embedding model/dim changes corpus-wide |

## Notes
- Needs `open-clip-torch` (pip). Weights (~350 MB) download to `~/.cache` on first run.
- Avenue/UCSD/Sky embeddings will cluster tightly (one scene each) — expected; they validate
  the conditioning plumbing, not text→content learning. See PROJECT.md 2026-07-31 decisions.
