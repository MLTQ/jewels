# `video_guide.py`

## Purpose

Aligns a low-resolution future video scaffold with the spatiotemporal birth grid. The oracle-guide
experiment uses true frames here; a successful result would later replace them with a pretrained
text-to-video model's coarse prediction.

## Components

### `video_to_cell_raster`
- **Does**: Resamples channel-last `(time,height,width,RGB)` frames to the configured `(u,v,t)`
  raster and flattens them in canonical cell order.
- **Interacts with**: `GridSpec` and the optional guide encoder in `BirthMarkFlowModel`.
- **Rationale**: An explicit order conversion prevents video `(t,v,u)` axes from being silently
  interpreted as jewel `(u,v,t)` axes.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Oracle-guide trainer/renderer | Output is `(spec.n_cells,3)` in `GridSpec` order | Axis order |
| Future video prior | RGB guidance can be produced at any positive source resolution | Input layout |

## Notes

- This intentionally contains no video decoder or model-specific normalization.
