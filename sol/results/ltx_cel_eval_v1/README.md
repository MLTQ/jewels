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

The matched reconstruction run completed on Aine using the RTX 2070S for Basketball/HorseRiding
and the RTX 4090 for PlayingGuitar/ApplyEyeMakeup. Every clip uses exactly 9,000 initial jewels,
spatial densification to 72,000, 9,000 optimizer steps, seed 0, and the same 49-frame geometry and
rendering contract as the photoreal gate. Checkpoints remain at
`/home/m/jewels/corpus/ltx_cel_eval_v1_72k`.

### Result

| Mean over four classes | Cel shaded | Photoreal control | Cel difference |
|---|---:|---:|---:|
| Full-volume PSNR | 28.675 dB | 31.472 dB | -2.797 dB |
| SSIM | 0.9616 | 0.9906 | -0.0290 |
| Flat-region RGB MAE | **0.00818** | 0.01272 | **-35.7%** |
| Contour-region RGB MAE | 0.05559 | **0.03424** | +62.3% |
| Contour / flat error | 7.14x | 2.66x | +4.47x |
| Edge-energy ratio (target 1.0) | 1.006 | 0.939 | +0.067 |
| Quiet temporal-change MAE | **0.00712** | 0.00847 | **-15.9%** |
| Target temporal change | 0.01529 | 0.03534 | -56.7% |
| Effective contributors / frame | 5,941 | 5,966 | -0.4% |
| Contributors above 5% peak alpha / frame | 7,163 | 6,883 | +4.1% |

| Class | PSNR | SSIM | Effective contributors / frame |
|---|---:|---:|---:|
| Basketball | 26.520 dB | 0.8978 | 7,544 |
| HorseRiding | 28.213 dB | 0.9660 | 4,240 |
| PlayingGuitar | 29.026 dB | 0.9882 | 6,174 |
| ApplyEyeMakeup | 30.940 dB | 0.9943 | 5,807 |

The broad-fill hypothesis passes: all four cel clips have lower flat-region error than their
class-matched photoreal controls. The whole-domain hypothesis fails under the current primitive
allocation because every class has higher contour-region error. Bold lines occupy little area but
carry most object identity, so uniform pixel loss and ordinary spatial splitting undersupply the
narrow moving curves. Near-perfect aggregate edge energy does not contradict this: blurred halos
and displaced double edges preserve the amount of edge energy without preserving its location.

The density audit rules out a simple count explanation. Cel and photoreal fields have effectively
the same contributors per frame, and the cel fields have more contributors above 5% peak alpha.
Temporal ratios above one are also easy to misread because the styled targets contain 57% less
motion; their absolute quiet-region temporal error is modestly lower.

### Decision

Do not begin style-specific learned-generator training on these fields as evidence that the current
jewel representation prefers animation. First run a fixed-72k fill/contour allocation control:
broad long-lived jewels for region interiors and narrow motion-aligned jewels for ink curves, with
edge-aware sampling or densification. The style remains a promising product wedge if that control
retains the 35.7% flat-region advantage while removing the 62.3% contour penalty.

`fit_gate_72k/contact_sheet.png` stacks all four source/reconstruction/error audits. Each class
folder contains the full 49-frame comparison GIF, contact sheet, and exact-render report.
`cel_metrics.json`, `photoreal_metrics.json`, `density.json`, and `summary.json` retain the complete
numeric evidence.
