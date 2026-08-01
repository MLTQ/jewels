# model.py

## Purpose
The jewel emitter: a DiT-style transformer velocity field for conditional flow matching over
primitive sets.

## Components

### `SetDiT`
- **Does**: (B, N, 22) noisy sets + flow time + optional CLIP embedding -> velocity
- **Rationale — NO positional encodings, anywhere.** Tokens are unordered, so the network is
  permutation-equivariant and the modeled distribution permutation-invariant. This is the
  architecture the canonicalization data demanded: primitives don't correspond across fits,
  distributions do. Adding any token ordering would silently reintroduce the gauge problem.
- Conditioning via adaLN-Zero (DiT): timestep embedding + projected CLIP vector; a learned
  `null_cond` replaces the CLIP vector under dropout -> classifier-free guidance (and thus
  t2v prompting through the shared CLIP space) is a sampling-time option, not a retrain.

### `Block`
- **Does**: pre-norm attention + MLP, adaLN-Zero modulation (gates init to zero)
- **Rationale**: zero-init gates + zero-init output proj make the initial velocity field ~0,
  which stabilizes early flow-matching training.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `cli/train_prior.py` / `cli/sample_prior.py` | forward(x, t, c, drop) -> same-shape velocity | signature; meta["model_args"] must reconstruct the module |

## Notes
- v0 scale: dim 256 / depth 6 / heads 8 ≈ 5M params — sized to memorize a 231-set corpus,
  which is the point of v0 (prove sample -> decode -> render end to end).
- Attention is SDPA; 6471 tokens x batch 4 fits comfortably on the 4090.
