# `plot_appearance_contract_evidence.py`

## Purpose

Compares the frozen bounded control/midpoint frontier with the matched residual-control and raw-
response experiment. It keeps sampled screens, exact metrics, structure gates, and per-style causal
deltas separate so the large representation jump is not confused with the smaller response gain.

## Components

### `load_evidence`
- **Does**: Maps the bounded audit's seed 0/1 to control/midpoint and the residual audit's seed 0/1
  to residual-control/raw-response.
- **Does**: Joins final sampled summaries with exact macro metrics and computes per-style response
  PSNR and lower-is-better LPIPS deltas against residual-control.

### `main`
- **Does**: Draws sampled/exact PSNR, LPIPS, SSIM, residual structure gates, and the per-style joint-
  improvement quadrant in one reproducible figure.
- **Does**: Shows the absolute `20 dB / 0.40 LPIPS` lines without treating them as achieved.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Appearance-contract report | Screen directories and audit candidate order match the frozen protocols | Naming/order |
| Causal interpretation | Raw response deltas use residual-control, not the older bounded control | Baseline mapping |
| LPIPS plot | Positive delta means lower/better LPIPS | Sign convention |
