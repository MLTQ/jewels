# Frozen geometry and residual appearance v1

## Outcome

Freezing the already-irregular time-distorted geometry removes the replication failure and lets
residual appearance cross exact 20 dB with ordinary additional compute. The selected two-tranche
render-only path reaches mean exact `20.05540 dB / 0.72270 LPIPS` across three continuation seeds.
Every seed exceeds 20 dB, every seed improves both PSNR and LPIPS over the original source in all
five held-out styles, and all 20 geometry state tensors remain bitwise identical to the source.

This is the strongest reconstruction-mechanism result so far. It establishes that a fixed sparse,
irregular, spacetime-tilted Gaussian field can support a replicated 20 dB amortized appearance
model. It does **not** meet the final `LPIPS <=0.40` representation gate, prove independent upstream
geometry replication, or prove text-conditioned generation.

The registered low-resolution full-frame perceptual objective is rejected: it loses both PSNR and
LPIPS to matched render-only continuation in all five styles. Its stability regularizers do reduce
temporal deviation, rendered range violations, and residual Jacobian energy, but trade away a small
amount of fidelity. Successes and failures are therefore recorded separately.

## Source and freeze proof

The source is the official residual seed-0 checkpoint from `appearance_contract_v1`, exact
`19.66477 dB / 0.73283 LPIPS`, with occupancy near `0.9846`, active fraction `0.62815`, and mixed
spacetime tilt near `0.521`.

`geometry_exact.json` directly compares every `geometry_trunk.*` and `geometry_head.*` tensor in
the three final checkpoints against that source:

- all three candidates: `bitwise_exact=true`;
- maximum absolute change: exactly `0`;
- tensors compared per candidate: `20`;
- mismatched tensors: none.

Every training evaluation independently snapshots and compares held-out center, log-scale,
quaternion, and opacity-logit predictions. All five sources in every run also report bitwise
equality. Structure cannot drift back toward a grid in this experiment.

## Read-only calibration and registered arms

Calibration used one declared training clip, 1,024 sampled points, and four contiguous `48 x 72`
frames without constructing an optimizer. The sampled render gradient norm was `0.01913`; grid RGB,
spatial, temporal, and structure norms were `0.17381`, `0.04403`, `0.06587`, and `0.85451`. This set
the intermittent full-frame term near half the sampled-render gradient in expectation rather than
choosing its weight from outcomes.

Three 600-step seed-0 arms started from the identical frozen source:

| Arm | Exact PSNR | LPIPS | SSIM | Layout PSNR |
|---|---:|---:|---:|---:|
| source | 19.66478 | 0.73288 | 0.82256 | 23.04958 |
| frozen-render | **19.89236** | **0.72605** | **0.83372** | **23.49562** |
| frozen-perceptual | 19.86420 | 0.72712 | 0.83228 | 23.43725 |
| frozen-stabilized | 19.85251 | 0.72836 | 0.83072 | 23.41652 |

Frozen-render beats frozen-perceptual on both PSNR and LPIPS in every held-out style. Relative to
render-only, perceptual loses `0.02816 dB` and worsens LPIPS by `0.00108`; stabilized loses
`0.03986 dB` and worsens LPIPS by `0.00231`.

The stability intervention is nevertheless causal within its registered tolerance:

| Arm | Mean absolute temporal-ratio error | Sampled RGB out of range | Residual Jacobian energy at final log |
|---|---:|---:|---:|
| frozen-render | 0.79196 | 4.282% | 0.03450 |
| frozen-perceptual | 0.78434 | 3.952% | 0.03255 |
| frozen-stabilized | **0.77793** | **3.605%** | **0.00812** |

It reduces the intended artifacts but does not supply the missing perceptual signal. The visual
difference is subtle and cannot justify selecting a dominated arm.

## Selected compute continuation

Frozen-render received one preregistered equal 600-step continuation with the same restarted AdamW
and cosine schedule semantics used earlier:

| State | Exact PSNR | LPIPS | SSIM | Layout PSNR |
|---|---:|---:|---:|---:|
| original frozen source | 19.66478 | 0.73288 | 0.82256 | 23.04958 |
| +600 frozen appearance | 19.89236 | 0.72605 | 0.83372 | 23.49562 |
| +1,200 frozen appearance | **20.08199** | **0.72206** | **0.83926** | **23.88868** |

The final state improves both PSNR and LPIPS over the original source in all five styles. Exact
PSNR gains range from `+0.2304` to `+0.8171 dB` in the shared final audit; LPIPS improvements range
from `0.00350` to `0.01587` for seed 0. Coarse object/person cues and boundary contrast improve
incrementally, while dark glitter remains visible. There is no qualitative collapse.

## Three-seed replication

Seeds 1 and 2 repeat the complete two-tranche selected path from the same upstream source. These are
appearance-optimization seeds, not independently trained geometry encoders.

| Continuation seed | Exact PSNR | LPIPS | Geometry exact |
|---:|---:|---:|:---:|
| 0 | 20.08200 | **0.72201** | yes |
| 1 | 20.03546 | 0.72300 | yes |
| 2 | 20.04873 | 0.72309 | yes |
| mean | **20.05540** | **0.72270** | yes |

PSNR range is only `0.04654 dB`; LPIPS range is `0.00108`. Every seed improves both metrics over
source in anime, cartoon, clay, photoreal, and render3d. The replicated sheets share the same coarse
content improvement and the same remaining blur/speckle family; there is no seed-specific collapse.

The generic gate inside `audit_irregular_encoder.py` still reports false because it averages the
source together with candidates and retains the final `LPIPS <=0.40` rule. The registered protocol's
per-seed 20 dB replication rule passes; the final representation gate does not.

## Visual evidence

- `evidence.png`: labeled compute curve, three-seed crossing, negative perceptual ablation, and
  partial stability result.
- `audit_seed0/qualitative.png`: source/render/perceptual/stabilized contact sheet.
- `continuation/audit_render_source_600_1200/qualitative.png`: selected compute continuation.
- `replication/audit_final_seeds/qualitative.png`: source and all final seeds.
- Every audit directory also contains `comparison.png`, `field_layout.png`, and its complete
  `report.json`.

The field-layout images remain visibly irregular in XY and X-time. They are a direct answer to the
earlier grid-noise concern: remaining visual granularity comes from appearance/compositing, not
quantized center placement.

## Feasibility assessment and next gate

This result materially advances the feasibility case. The project now has a compute-scaled,
three-seed, exact-audited 20 dB reconstruction result while preserving the defining irregular,
time-distorted Gaussian geometry bit for bit. That is credible evidence that the representation is
not blocked at the previous fidelity threshold.

The pitch is still premature because LPIPS remains about `0.723`, far from `0.40`, and the field is
visually much blurrier than the lattice (`0.428`) or fitted ceiling (`0.172`). More render-only
compute is improving LPIPS slowly; extrapolating it alone is not credible.

The next experiment should keep the proven geometry and 20 dB base frozen, then add a zero-expanded
high-resolution appearance adapter trained on train-only saliency/edge patches with direct feature-
perceptual supervision. The adapter should have finer spatial evidence than the current single-point
32-channel appearance sample, and range/stability penalties should remain small guardrails rather
than the main objective. A useful next gate is replicated exact PSNR `>=20`, LPIPS `<0.70`, lower
out-of-range/temporal error, and visibly sharper actor/object boundaries before text-prior work.

## Execution integrity

`EXECUTION_NOTE.md` records GPU UUID ownership, the service restart, one seed-2 OOM before its first
optimizer step, and the authorized fallback to the available 4090. Final seeds were evaluated
together in one process; runtime and device are not used as model evidence.
