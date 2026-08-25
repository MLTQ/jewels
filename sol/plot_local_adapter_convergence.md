# `plot_local_adapter_convergence.py`

## Purpose

Builds the compute-curve evidence for frozen-base local appearance adapters. It keeps noisy
training diagnostics, fixed sampled validation, and shared exact seven-frame audits visibly
separate so a short screen cannot be mistaken for convergence.

## Components

### `collect_convergence`
- **Does**: Joins the raw-local 12k run and its 4k continuation, both independent scale-32
  derivative runs, the original 400-update screen, and exact milestone audits under semantic arm
  labels.
- **Does**: Reads fixed validation metrics from the nested `evaluation` records written beside
  training aggregates in each JSONL log.
- **Rationale**: Generic audit candidate labels are positional; explicit milestone mappings retain
  the actual experimental provenance.

### `plot_convergence`
- **Does**: Draws smoothed training LPIPS, fixed validation PSNR, exact LPIPS, and exact PSNR curves.
- **Does**: Marks the registered LPIPS `<0.70` pitch gate and `20 dB` reconstruction floor.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Convergence report | Raw continuation steps are offset by the initial 12,000 updates | Run schedule or path changes |
| Scientific review | Training, nested sampled-validation, and exact audit metrics remain labeled separately | Metric provenance |
| Plot tests | Dense seed-0 and independent seed-1 audit candidate ordering matches the registered milestones | Audit ordering |
