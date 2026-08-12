# `compare_realizer_reports.py`

## Purpose

Produces a matched, macro-by-source comparison of video-to-jewel realizer visual reports so an
architecture is not selected from one favorable metric or class.

## Components

### `compare_realizer_reports`

- **Does**: Requires identical unique source sets, compares one named panel, and reports baseline,
  candidate, and signed deltas for PSNR, SSIM, contrast, edge, saturation, temporal change, and all
  topology-adherence metrics.
- **Interacts with**: `render_prompted_mark_flow.py` JSON reports.
- **Rationale**: Render supervision can legitimately trade paired PSNR for detail energy. Reporting
  the complete vector and every per-source delta prevents a lateral trade from being called a
  general improvement.

### `main`

- **Does**: Reads two JSON reports, prints the comparison, and optionally writes it to a durable
  result artifact.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Ablation reports | Source IDs and selected panel exist in both inputs | Input schema |
| Paper tables | Aggregate values are arithmetic means over matched sources | Aggregation policy |
| Visual gate | Signed per-source deltas accompany macro metrics | Output schema |

## Notes

- Higher is not automatically better for target-relative ratios: contrast, edge, saturation, and
  temporal change ideally approach `1.0`, so direction and absolute target error must both be read.
