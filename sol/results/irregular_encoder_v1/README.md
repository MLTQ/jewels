# Irregular jewelfield feasibility report

- **Status:** geometry hypothesis supported; promotable text-to-video proof not yet achieved
- **Branch:** `codex/irregular-jewelfield-gate`
- **Evaluation:** five held-out cooking videos, one per style; seven fixed frames per source
- **Compute:** local GPU host, bounded screens followed by selected 6k-step runs

## Bottom line

The observed grid is architectural, not numerical quantization. The frozen baseline creates a
`16 x 16 x 8` cell lattice with 36 slots per cell (73,728 always-active proposals), samples RGB at
those locations, and limits learned centre motion to 0.75 cell. Its 36-slot initializer also has a
specific defect: `round(cuberoot(36)) == 3`, so slots 27–35 duplicate slots 0–8 exactly.

The new encoder fixes that initializer, permits four-cell centre migration, samples colour at the
predicted continuous centres, uses quaternion anisotropic covariances, and makes opacity sparsity
trainable. Its centre plot is genuinely irregular in both XY and XT. The residual visual defect is
blur from too few/broad appearance primitives, not a hidden position grid.

This is meaningful progress, but it is not yet the pitch proof. The best geometry-preserving field
passes every structural threshold and fails both appearance thresholds. More steps alone improve
fidelity while pulling centres back toward uniform coverage; freezing geometry prevents that drift
but exposes a saturated appearance head. The next experiment must increase and decouple appearance
capacity, not merely extend the current run.

## Decisive held-out audit

| Arm | PSNR dB ↑ | LPIPS ↓ | Occupancy uniformity ↓ | Active fraction ↓ | Mixed tilt ↑ |
|---|---:|---:|---:|---:|---:|
| Frozen lattice baseline | 23.868 | 0.392 | 0.99919 | 1.000 | 0.248 |
| Joint direct-tilt, step 6k | 20.002 | 0.774 | 0.99017 | 0.617 | 0.521 |
| Geometry frozen at step 2k + 4k appearance | 19.092 | 0.799 | **0.98070** | **0.664** | **0.507** |
| Independently fitted teacher ceiling | **27.699** | **0.172** | — | — | — |

The frozen candidate passes occupancy `<= 0.985`, active fraction `<= 0.70`, mixed tilt `>= 0.25`,
and “less uniform than lattice.” It misses PSNR `>= 20` and LPIPS `<= 0.40`. The joint 6k arm
barely passes PSNR but misses occupancy and LPIPS. No candidate passes the preregistered gate.

## What the images establish

- [`field_layout.png`](audit_direct_tilt_seed0_step6000/field_layout.png) plots active centre
  locations only. The baseline is a repeated spatial/time grid; the candidate is irregular; the
  fitted teacher is content-clustered. This directly answers the quantization question.
- [`qualitative.png`](audit_frozen_geometry_seed0_total6000/qualitative.png) compares target,
  lattice, frozen irregular field, and fitted ceiling. The irregular arm loses the checker pattern
  but is visibly under-detailed and broad.
- [`comparison.png`](audit_frozen_geometry_seed0_total6000/comparison.png) places the exact audit
  against the frozen thresholds.
- [`progression.png`](progression.png) shows the compute trajectory and the empty desired quadrant:
  no tested arm is simultaneously above 20 dB and below 0.985 occupancy at its final checkpoint.

## Experiment ledger

Claims below are deliberately scoped. A failed arm rejects only its declared settings.

| Experiment | Held-out result | Supported conclusion |
|---|---|---|
| Immediate sparsity, 200 steps | 16.085 dB, 0.9788 occupancy, 0.540 active | Sparsity causally removes the grid early, with an early fidelity cost. |
| Matched no-sparsity control, 200 steps | 17.989 dB, 0.9986 occupancy, 1.000 active | Render loss alone prefers uniform coverage. |
| Appearance-biased, 12 teachers, 2k→4k→6k | 19.679→20.809→21.261 dB; occupancy 0.9861→0.9876→0.9894 | Compute improves fidelity, but the current objective gradually re-grids the field. |
| Stronger density, 12 teachers | Early occupancy 0.9737; final 20.035 dB/0.9886 | Density pressure can de-grid; this schedule does not preserve it through fidelity training. |
| Teacher coverage expanded 12→36 | 600-step 19.232 dB/0.9774 occupancy | Broader successful supervision justified the selected full run; it does not prove scaling alone. |
| Strong cosine orientation surrogate | Mixed tilt 0.084 | This surrogate is insufficient; tilted jewels remain viable. |
| Direct tilt supervision, 600-step screen | 18.953 dB, 0.9723 occupancy, 0.672 active, 0.515 tilt | Direct optimization learns the intended time-distorted geometry. |
| Direct tilt, 6k | 20.131 dB sampled evaluation; exact audit above | The same joint head recovers fidelity while structure drifts. |
| Exact frozen geometry + 4k appearance | Geometry unchanged at every checkpoint; sampled PSNR 19.048→19.305, exact audit 19.092/0.799 LPIPS | Existing colour/linear-gradient/background outputs saturate; longer training of those rows is not enough. |
| Initial support capacity 4,096 | Overflow at step 121; rerun at 8,192 completed | A renderer-capacity setting failed, not the representation. |

The raw compact logs are in [`evidence_logs`](evidence_logs/); exact per-source audit records and
gate booleans are in each audit directory's `report.json`.

## Feasibility assessment

### Demonstrated

1. Support-correct time-distorted Gaussian splats can fit these videos at 27.70 dB / 0.172 LPIPS.
2. A feed-forward encoder can emit sparse, mobile, non-lattice centres.
3. It can learn mixed spacetime axes when the actual property is supervised directly.
4. Held-out fidelity has a positive compute slope under the appearance-biased objective.

### Not demonstrated

1. One amortized checkpoint has not passed structure and appearance gates together.
2. The irregular result has not replicated across three seeds.
3. No text-conditioned prior has yet generated the irregular latent on held-out prompts.
4. Therefore “more compute will produce a promotable T2V model” is not yet a defensible claim.

The representation remains feasible; the current coupled 20,480-proposal decoder is the failed
unit. The fitted ceiling and irregular-centre success argue for another architecture iteration.

## Next experiment

Build `structural_jewel_encoder_v3` with an explicitly factorized decoder:

1. **Geometry branch:** centres, quaternion/scales, and opacity. Train to the successful structural
   gate, then freeze or stop-gradient it.
2. **Appearance branch:** sample a multiscale feature pyramid at the predicted continuous centres
   and use a deeper per-jewel MLP for colour and higher-order colour basis. It must not receive a
   gradient path that can regularize centre positions back into coverage.
3. **Capacity sweep:** compare 20,480, 40,960, and 73,728 proposals at matched steps. The successful
   frozen field uses only 13,595 active jewels versus 73,728 in the lattice and fitted ceiling;
   current blur is confounded with a 5.4× active-capacity gap.
4. **Appearance objective:** multiscale reconstruction plus a bounded perceptual term, selected on
   validation LPIPS rather than sampled-voxel PSNR alone.
5. **Promotion rule:** only after one setting passes the unchanged gate, replicate seeds 0/1/2.
   Then train a small text/label-conditioned latent prior and require held-out prompt motion to beat
   a static-video baseline before preparing the pitch.

The fastest informative screen is the factorized 40,960-proposal arm initialized from continuous
stratified centres, with geometry stopped after it passes occupancy/tilt. It directly tests whether
the empty upper-left quadrant in `progression.png` is a capacity/coupling problem.
