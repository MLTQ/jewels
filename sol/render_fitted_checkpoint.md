# `render_fitted_checkpoint.py`

## Purpose

Renders an already fitted corpus checkpoint against the exact source window without repeating the
expensive optimization run.

## Components

### `main`

- **Does**: restores fit configuration, field, background, and source window; computes true
  full-volume PSNR; writes a comparison GIF, contact sheet, and JSON report
- `--video` and `--start-frame` can supply provenance for controlled experiment checkpoints that
  predate embedded `source` metadata. An explicit override is recorded in the output report.
- **Interacts with**: `stprim/cli/render_recon.py`, `PrimitiveField`, and `load_video`
- **Rationale**: density sweeps must compare identical render/metric code and must never refit merely
  to obtain visual artifacts

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Density experiments | Checkpoint carries `state`, `cfg`, `info`, and `source` | Corpus schema |
| Visual review | GIF is target/reconstruction; contact rows are target/reconstruction/error×5 | Layout |

## Notes

- The source video must remain at the path stored in the checkpoint, or be passed explicitly with
  `--video` for experiment checkpoints that do not carry a `source` key.
- Full-volume PSNR uses the fitter's kNN setting and learned background.
