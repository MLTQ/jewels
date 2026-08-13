# Cel-shaded LTX fixed-density gate

This experiment tests whether a low-texture rotoscoped/cel-shaded video language is a better match
for spacetime Gaussian jewels than the photoreal LTX controls. It uses the same four held-out action
prompts and seeds as the existing LTX evaluation set; only the generation suffix changes.

## Generated corpus

| Measurement | Result |
|---|---:|
| Completed / failed | 4 / 0 |
| Geometry | 768x512, 49 frames, 24 fps |
| Aggregate / mean generation time | 748.97 / 187.24 s |
| Peak GPU memory | 9,408--9,409 MiB |
| Generation GPU | RTX 4090 |

The visual-language suffix requests rotoscoped cel shading, broad flat color regions, a restrained
palette, stable ink contours, minimal texture, clean color holds, continuous motion, and no cuts.
The source prompts remain separately recorded in `manifest.json`; the original evaluation seeds
are 42003, 42103, 42203, and 42303.

The visual gate passes. Basketball is nearly monochrome line animation; HorseRiding combines broad
gray fills with heavy contours; PlayingGuitar uses a small red/cream/yellow palette; and
ApplyEyeMakeup is dominated by flat skin, hair, and background regions. The combined sheet samples
frames 0, 16, 32, and 48 in class order.

- `generation_contact.jpg`: four-class source audit.
- The four MP4 files are the exact generated sources used by fitting.
- `generation_receipts/`: per-sample commands, upstream revision, GPU telemetry, and media probes.
- `manifest.json`: prompt/seed/runtime identities and completion summary.

## Frozen jewel-fit gate

The matched reconstruction run started on Aine's RTX 2070S as
`jewels-cel-fit-v1.service`. It fits one 49-frame window per clip using exactly 9,000 initial
jewels, spatial densification to 72,000, 9,000 optimizer steps, seed 0, and recovery every 100
steps. Checkpoints are written to `/home/m/jewels/corpus/ltx_cel_eval_v1_72k`.

The decision metric is not style appearance alone. Completed fits will be compared with the
photoreal four-class gate at identical budget using full-volume PSNR/SSIM, edge and temporal-change
retention, palette stability, and contribution-aware jewels per frame.

