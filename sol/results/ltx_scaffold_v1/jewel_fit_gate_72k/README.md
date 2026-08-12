# Density-matched LTX jewel-fit gate

This folder records the four optimized jewel reconstructions of LTX-2.3 prompt-generated
evaluation scaffolds. Basketball was the calibration sample; HorseRiding, PlayingGuitar, and
ApplyEyeMakeup then used the frozen contract unchanged.

## Frozen contract and Basketball calibration

| Measurement | Result |
|---|---:|
| Source / fitted geometry | 49 frames, 160×240 |
| Raw jewels | 72,000 |
| Fit schedule | 9,000 steps; 9,000 initial; 9,000-step spatial growth ceiling |
| Fit time on RTX 4090 | 1,523.1 s (25.4 min) |
| Full-video PSNR | 28.182 dB |
| Mean / range effective contributors per frame | 6,384 / 5,485–7,626 |
| Mean / range contributors above 5% peak alpha | 7,389 / 6,463–8,828 |

The 72k budget scales the established 120k-jewel, 96-frame contract by spacetime volume rather
than treating total jewel count as resolution-independent. Its measured effective density lands in
the target 5k–10k contributor/frame regime.

Across all four classes, mean replay PSNR is 31.472 dB, opacity-weighted effective density is 5,966
contributors/frame, and 6,883/frame exceed 5% potential peak alpha.

| Class | Full-video PSNR | Effective / frame | Above 5% alpha / frame |
|---|---:|---:|---:|
| Basketball | 28.182 dB | 6,384 | 7,389 |
| HorseRiding | 31.707 dB | 7,134 | 8,042 |
| PlayingGuitar | 30.839 dB | 4,597 | 5,339 |
| ApplyEyeMakeup | 35.158 dB | 5,749 | 6,762 |

## Visual interpretation

The fit preserves the outdoor court, player layout, basketball action, and red uniforms across the
window. It visibly softens fine fence/tree detail and fast limbs. This passes as a coherent
semantic scaffold and video-to-jewel target; it does not establish final codec quality.

- `contact_sheet.png`: source row, jewel reconstruction row, and absolute error ×5 at frames
  0/12/24/36/48.
- `compare.gif`: source and reconstruction through all 49 frames.
- `report.json`: full-volume render provenance and PSNR.
- `ltx_scaffold_v1_eval_72k_basketball_density.json`: contribution-aware per-frame density.

The authoritative checkpoint remains on Aine at
`/home/m/jewels/corpus/ltx_scaffold_v1_eval_72k/00_basketball_evaluation_00_seed42003_w000000.pt`.

## Horse Riding confirmation

The second evaluation-prompt fit independently confirms the density contract. It trains for
1,522.2 seconds and replays exactly at 31.707 dB with 7,134 effective contributors/frame on
average (4,803--8,064). The source/reconstruction/error contact sheet and GIF are under
`01_horseriding/`. Rider, horse, fence geometry, and the rider's blue shirt remain stable; most
visible error follows fine edges and moving limbs.

## Playing Guitar: density is a diagnostic, not a quota

The third fit trains for 1,519.8 seconds and replays exactly at 30.839 dB. Its mean count above 5%
peak alpha is 5,339/frame, inside the intended band, while the stricter opacity-weighted effective
count is 4,597/frame (2,994--5,525), just below the nominal floor. Visual inspection nevertheless
preserves the red guitar, white pickguard, hands, neck, and lighting very well. The evidence argues
for content-aware density control and visual/quality gates, not automatically adding primitives
until every density statistic crosses one scalar threshold. Artifacts are under `02_playingguitar/`.

## Apply Eye Makeup confirmation

The final fit replays at 35.158 dB. It averages 5,749 effective contributors/frame
(4,295--6,449), while 6,762/frame exceed 5% peak alpha (4,920--7,536). Face identity, skin tone,
white shirt, hand, cosmetic tool, and eye-level motion remain stable. Its optimizer history reports
3,650.1 seconds after a checkpoint-preserving migration from the RTX 4090 to the RTX 2070 Super;
that mixed-device time is not directly comparable to the three uninterrupted 4090 fits. Artifacts
are under `03_applyeyemakeup/`.

`ltx_scaffold_v1_eval_72k_all_density.json` is the authoritative four-class density audit.
