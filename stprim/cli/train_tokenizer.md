# train_tokenizer.py

## Purpose
Train the jewel tokenizer end-to-end on a fitted corpus; finish with the round-trip render
PSNR that answers whether the latent bottleneck can carry a scene.

## Components

### `main()`
- **Does**: load corpus -> standardize (stats into checkpoint meta, mu stats into encoder
  buffers) -> joint encoder+decoder flow-matching loop (bf16, warmup+cosine) -> save ->
  encode window 0, sample decoder back, render both, report PSNR
- **Interacts with**: `prior/tokenizer.py`, `prior/featurize.py`, `models/render.py`

## Decisions
- Runs on the 2070 (`--device cuda:1`) while the 4090 fits the dense corpus — the linear-in-N
  design is what makes that possible.
- Developed/validated against the 10k-era corpus first; retrained on the dense corpus when it
  lands. Architecture is N-agnostic, so this is a config change, not a rewrite.
- Round-trip PSNR compares tokenized render to FITTED render — the tokenizer's job is to
  preserve the fit, not to beat it.

## Notes
- ~17× compression at N=6471 with default grid 8×8×4 × latent 32.
