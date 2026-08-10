# Oracle-topology stochastic mark-flow control

This 12,000-step, 1.54M-parameter rectified-flow run replaces deterministic 22-D mark regression
while holding target birth cells, counts, and ranks exact. It asks whether sampling rather than
conditional averaging restores detail.

Raw held-out samples average 15.064 dB, 0.775 target contrast, and 0.844 target edge energy, versus
14.232 dB, 0.574, and 0.598 for deterministic oracle-topology marks. The images are sharper but
remain incoherent speckle. Stochastic set generation is necessary; local prefix statistics and a
four-class prompt do not provide enough global spatial semantics.

- `summary.json` and `train_log.jsonl` record the completed fixed-path run.
- `visual/mark_flow_visual_report.json` contains all held-out measurements.
- `visual/*_mark_flow_controls.gif` and contact sheets compare deterministic, stochastic,
  shuffled-text, and text-only outputs.
