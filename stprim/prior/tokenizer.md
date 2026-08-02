# tokenizer.py

## Purpose
The O(N²) wall-breaker and the vocabulary thesis in one module: set of N jewels <-> ~256
latent tokens, both directions LINEAR in N. Gate for training any prior on the dense
(45k-jewel) corpus; later the VQ variant is the codec entropy model.

## Components

### `GridPoolEncoder`
- **Does**: bucket jewels into a coarse (u,v,t) grid (default 8×8×4=256 cells), mean-pool
  per cell, transformer over CELLS, project to latent_dim per cell
- **Rationale**: linear in N; permutation-invariant because bucketing ignores order. The
  cell embedding is structure over SPACE, not token order — it does not reintroduce the
  ordering gauge the canonicalization data forbids. Features arrive standardized, so
  `mu_mean/mu_std` buffers de-standardize position for bucketing (set by the train script).

### `LatentSetDecoder`
- **Does**: conditional flow velocity v(x_t, t | latents); each jewel cross-attends to
  latents, NEVER to other jewels
- **Rationale**: linear in N. Inter-jewel coordination must route through the latents —
  that is precisely the compression bet: if 256×32 numbers can't coordinate a scene, the
  round-trip PSNR says so immediately.

### `JewelTokenizer.flow_loss / reconstruct`
- **Does**: end-to-end training objective (conditional flow matching) and the sampled
  round-trip
- **Rationale**: flow matching IS the permutation-invariant reconstruction loss — no
  Chamfer, no Hungarian matching, and the decoder doubles as the generative decode stage
  for the latent prior.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| latent prior (next) | encoder output (B, n_cells, latent_dim); decoder conditioned on same | latent shape/meaning |
| codec angle (future) | VQ variant of the latent space | — |

## Notes
- v0 latents are continuous; VQ lands after reconstruction is proven.
- Compression at N=6471: 142k numbers -> 8192 (~17×). At N=45k: ~126×.
- The metric that matters is round-trip RENDER PSNR (train_tokenizer prints it), not flow
  loss — flow loss has the same irreducible floor as the prior's.
