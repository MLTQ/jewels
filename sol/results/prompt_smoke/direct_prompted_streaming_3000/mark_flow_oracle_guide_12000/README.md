# Oracle-video-guided jewelizer control

This 12,000-step, 2.13M-parameter rectified-flow run receives the true future stride as a `24x40`
low-resolution raster aligned to the `16x16x8` birth grid. It retains exact target topology. The
guide is privileged information used to test architecture; this is not prompt-only inference.

Across four group-held-out UCF videos, guided projected samples average 16.555 dB, 0.856 target
contrast, and 0.905 target edge energy. Deterministic oracle-topology marks average 14.232 dB,
0.574, and 0.598. Thus the semantic scaffold yields +2.324 dB and recovers recognizable macro-layout
that the topology-only and unguided models do not. Remaining noise motivates multiscale guide
features, cross-attention, and direct render/perceptual supervision.

`visual_contract_projection/` is authoritative. Its hard projection preserves target edge cells
and discrete temporal birth bins; projecting a fitted target is render-identical at 100 dB.
`visual/` is retained as provenance from the preliminary continuous-bound projection and should not
be used for quantitative comparison.

- `summary.json` and `train_log.jsonl` record training.
- `visual_contract_projection/mark_flow_visual_report.json` contains final metrics.
- The GIFs and contact sheets compare target, deterministic marks, zero-guide flow, guided raw and
  projected flow, and shuffled text under the same oracle guide.
