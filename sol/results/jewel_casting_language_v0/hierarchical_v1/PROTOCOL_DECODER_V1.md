# Hierarchical phrase decoder Gate 0e protocol

## Decision question

Can a learned decoder recover the missing continuous correlations from the stable hierarchical
phrase without receiving the true residual? Gate 0d reached 19.06 dB token-only on nine fresh
source-disjoint fields while passing canonicality, tilt, irregularity, and decision-cost checks.
Gate 0e changes only the decoder from fixed prototype composition to a learned product-code head.

## Frozen inputs and ownership

- Frozen Gate-0d pair and individual K=1,024 codebooks.
- The same 33 source-disjoint training fields and nine fresh validation fields.
- Per pair input: pair layout/covariance IDs, two individual surface IDs, two individual gradient
  IDs, cell, exact irregular pair anchor, and count.
- Forbidden forward inputs: target bundle values, exact residuals, source pixels, video latents, or
  validation-source identity.
- Training samples at most 4,096 pair phrases per source, balanced across the 33 training sources.
- Exhaustive evaluation uses every pair in all nine validation fields.

## Frozen model and schedule

- Six independent 1,025-entry role embeddings (1,024 tokens plus padding), dimension 32.
- Cell embedding 32, count embedding 16, three-frequency Fourier features of the continuous anchor.
- Four-layer 384-wide SiLU MLP predicting a 2x22 correction around the frozen product prototypes;
  each correction is smoothly bounded to three train-owned RMS units.
- Train-owned per-row/per-feature RMS scales normalize the correction loss.
- AdamW, learning rate 3e-4, weight decay 1e-4, batch 2,048, seed 20260830.
- Maximum 15,000 updates; evaluate every 500. Stop only after ten evaluations without at least 0.1%
  relative validation improvement. Retain the best validation checkpoint.
- Validation rendering uses the same support-complete 4,096 source-owned random points as Gate 0d.

## Registered gate

All checks must pass:

1. Validation normalized correction MSE improves at least 10% over the frozen zero-correction
   product decoder.
2. Learned source-relative random-volume PSNR is at least 20 dB and improves at least 1 dB over its
   exactly matched raw product decode.
3. Learned median mixed spacetime tilt remains within 15% of the source.
4. Learned centers have less than 1% exact cell-center locking.
5. The saved architecture audit confirms `target_values` and exact residuals are loss-only and are
   not model forward inputs.

A pass licenses the free-running prompt-conditioned caster. A failure blocks that caster and calls
for a contextual within-cell decoder; it does not license target-residual leakage or validation
fine-tuning.
