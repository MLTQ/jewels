# Frozen render two-seed replication v1 — registered protocol

## Trigger

The selected seed-0 two-tranche path passes the continuation rule. Exact PSNR reaches `20.08199 dB`
and LPIPS falls to `0.72206`, versus original frozen source `19.66478 / 0.73288` and first frozen
tranche `19.89236 / 0.72605`. SSIM reaches `0.83926`, layout PSNR `23.88868`, and every style beats
the original source on both PSNR and LPIPS. Held-out geometry remains bitwise exact.

## Registered replication

- For continuation seeds 1 and 2, start from the same official residual source
  `appearance_contract_v1/continuation_2070/control_total1200/encoder.pt`.
- Run two consecutive 600-step frozen-render tranches per seed. Each tranche restarts AdamW and the
  600-step warmup/cosine schedule exactly as seed 0 did; the second initializes from that seed's
  first frozen checkpoint.
- Preserve the complete selected configuration: sampled render MSE only, geometry frozen, all
  grid/range/residual/teacher/local/structure/sparsity weights zero, 120-video manifest, peak LR
  `1e-4`, warmup 100, 1,024 points, support capacity 8,192, and five ordered validation sources.
- Pin all runs by hardware UUID to the physical RTX 2070 Super. Runtime is not model evidence.

## Decision rules

Each final seed must report held-out `bitwise_exact=true`, `max_abs_change=0`, occupancy `<=0.985`,
active fraction `<=0.70`, mixed tilt `>=0.25`, and finite sampled PSNR. All final seeds are then
exact-audited together with seed 0 under the same seven-frame LPIPS/PSNR/SSIM/layout protocol.

Replication passes only if every seed reaches exact PSNR `>=20 dB`, improves LPIPS over the original
source `0.73288`, and improves both PSNR and LPIPS over that source in all five styles. Report mean,
range, and every individual seed; a favorable mean cannot hide a failure. Visual review must show
the shared coarse-content gain without a seed-specific speckle collapse.

Passing establishes optimization-robust evidence that fixed irregular time-distorted geometry can
support a 20 dB residual appearance model with more compute. It does not establish the final
`LPIPS <=0.40` representation gate, independent upstream-geometry replication, or text-conditioned
generation.
