# Four-class shared-tokenizer smoke

## Protocol

This is the bounded 500-step trainability gate for one tokenizer shared across all 16 fitted prompt
windows. Training uses source groups 1--3 (12 windows); group 4 from every class remains entirely
held out. The model has a `32^3` grid, 512 slots/cell, 24 latent channels, rank-conditioned local
encoding, and sparse variable-count decoding. Three quarters of its sampled render points come from
deterministic source-video saliency pools split evenly between motion and saturated chroma; no
held-out source pixels enter those pools.

The architecture has 21.48M parameters and maps 990k raw jewel values to 786,432 latent values
(3.36x numeric compression). This is a research bridge to the already-validated `16^3` hierarchy,
not yet a practical language-token rate.

## Result

At 500 steps, the fixed 4,096-point audit over all four held-out sources reaches **15.890 dB
macro/mean PSNR** and **95.74% mean count recovery**, up from 12.007 dB and zero decoded jewels at
initialization.

| Held-out source | PSNR | Decoded / target jewels |
|---|---:|---:|
| ApplyEyeMakeup group 4 | 15.557 | 110,957 / 120,000 |
| Basketball group 4 | 15.620 | 116,554 / 120,000 |
| HorseRiding group 4 | 14.137 | 109,473 / 120,000 |
| PlayingGuitar group 4 | 18.245 | 122,571 / 120,000 |

The smoke therefore passes the structural gate: every class improves from initialization and count
recovery exceeds 95%. It does **not** pass the visual tokenizer gate. The animations retain broad
scene palette and some foreground motion, but actor geometry is diffuse and high-frequency sparkle
dominates the round-trip. Five hundred steps establish that the shared model trains; they do not
establish that the representation preserves prompt-relevant detail.

A fresh 3,000-step run with the same split and loss began on the RTX 2070 SUPER after this smoke.
Its decision gate is recognizable held-out actors and class-specific motion/color, plus a material
improvement over the 17.786 dB frozen cross-domain baseline. A numerical gain without a visual gain
is a no-go for expanding the corpus.

## Artifacts

- `heldout_eval_4096.json`: fixed high-sample audit used for the table
- `summary.json` and `train_log.jsonl`: training trajectory and built-in evaluation
- `manifest.json`: rendered-frame and decoded-count provenance
- `*_dense_roundtrip.gif`: fitted target on the left, shared-tokenizer round-trip on the right

The 258 MB research checkpoint remains on the compute host and is intentionally not tracked.
