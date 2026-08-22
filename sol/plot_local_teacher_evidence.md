# `plot_local_teacher_evidence.py`

## Purpose

Turns the preregistered local-teacher screen and exact-render audit into one pitch-readable figure.
The plot keeps sampled screening, exact perceptual metrics, and irregular-field constraints visibly
separate so a proxy improvement cannot be mistaken for an exact-render result.

## Components

### `load_screen_metrics`

- Reads only each arm's final held-out evaluation.
- Normalizes sampled PSNR, occupancy, active fraction, mixed spacetime tilt, and median extent.

### `load_exact_metrics`

- Applies the audit's frozen arm mapping: seed 0 is control, seed 1 is appearance-local, and seed 2
  is full-local.
- Returns macro exact-render metrics plus per-style PSNR deltas and LPIPS improvements relative to
  the control.

### `main`

- Draws the sampled eligibility band, absolute promotion gates, structural gates, and per-style
  exact-render improvement quadrant.
- Uses separate panels for PSNR and LPIPS because their units and preferred directions differ.
- Labels the load-bearing bars with their exact values so the figure remains useful outside the
  surrounding report text.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local-teacher report | Screen directories use `<arm>_seed0_600/summary.json` | Run naming |
| Exact audit | Candidate order is control, appearance, full | Audit CLI order |
| Gate interpretation | Thresholds match `PROTOCOL.md` | Protocol revision |
