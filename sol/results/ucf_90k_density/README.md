# UCF 90k isotropic-density control

## Question

Does doubling a 64-frame UCF fit from 45k to 90k jewels reach the intended 5k–10k useful splats
per frame and improve visible reconstruction quality under the existing isotropic split policy?

## Result

No. The fit completed safely, but raw count converted poorly into useful per-frame density.

| Measurement | 45k isotropic | 90k isotropic | Change |
|---|---:|---:|---:|
| Temporal 3σ intersections/frame | 5,546 | 7,202 | +29.9% |
| Peak alpha ≥1%/frame | 4,894 | 6,401 | +30.8% |
| Peak alpha ≥5%/frame | 3,595 | 4,724 | +31.4% |
| Effective peak-alpha contributors/frame | 2,933 | 3,822 | +30.3% |
| Full-volume PSNR | **34.136 dB** | 33.062 dB | **−1.074 dB** |
| Fit runtime | 56.8 min | 79.7 min | +40.3% |
| Mean finite-support lifespan | **8.41 frames** | 5.58 frames | **−33.6%** |
| Median finite-support lifespan | **5 frames** | 3 frames | **−40.0%** |
| Observed jewel births/frame | 659 | 1,290 | +95.6% |

The 90k fit remains below 5k effective contributors, and edge frames fall to 2,654 effective / 3,177
above 5% peak alpha. The matched visual A/B shows modest sharpening in some actor/edge regions but
lower aggregate faithfulness.

## Diagnosis

The legacy split shrinks every principal scale by 1.6. This shortens temporal lifespan as well as
spatial footprint, so doubling total jewels does not double active jewels per frame. It also cuts
the two-child integrated Gaussian volume to roughly `2 / 1.6³ = 0.488` of the parent before
re-optimization, explaining the repeated loss shocks and harder convergence.

The persistent-streaming audit makes the failure mode more explicit. In 16-frame strides, the 90k
fit carries only 6.2k–6.3k existing jewels across each internal boundary while emitting 20.5k–20.9k
new jewels in the next two interior strides. The 45k fit carries about 5.3k and emits 10.1k–10.5k.
The added capacity is therefore being spent predominantly on short-lived births rather than a denser
persistent state.

The deterministic carry/commit contract itself passes. Rendering each stride from only its carried
and newly born active jewels agrees with the monolithic finite-support render to `1.19e-7` max error
for 45k and `5.96e-8` for 90k, with every sampled point committed exactly once. This validates stable
ID/window ownership but does not validate learned continuation.

This matches current 4D Gaussian findings: short-lifespan/inactive Gaussians create redundancy, and
static densification rules under-refine dynamic regions. See [4DGS-1K](https://papers.neurips.cc/paper_files/paper/2025/hash/6c39d1a7eecfd98570d74bf7efec1be7-Abstract-Conference.html),
[SharpTimeGS](https://arxiv.org/abs/2602.02989), and
[Temporally Aware Densification](https://arxiv.org/abs/2606.23212).

## Decision

Do not scale the isotropic policy to 180k. The next matched control keeps the most time-aligned
principal scale, shrinks only the two spatial-like axes by √2, and rotates split jitter from local
principal coordinates into world `(u,v,t)` coordinates.

## Artifacts

- `source_45k_90k.gif`: source, matched 45k fit, matched 90k fit across all frames
- `source_45k_90k_contact.png`: frames 0/16/32/48/63
- `density_ucf_90k.json`: contribution-aware density report
- `streaming_contract_45k.json`: matched lifecycle, birth-rate, and carry/commit audit
- `streaming_contract_isotropic.json`: 90k lifecycle, birth-rate, and carry/commit audit
- `report.json`: full-volume 90k metric
- `baseline_45k/report.json`: matched 45k metric
