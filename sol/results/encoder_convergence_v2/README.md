# Balanced encoder convergence curve: first phase

This directory preserves the first 120 corpus passes for the corrected 12/60/120-example encoder
curve. The subsets are exact and nested. The 60-example set contains one item from every one of the
5-style × 12-action strata; the 120-example set contains two per stratum. All three seeds use the
support-complete tiled renderer and the same frozen 60-example validation inventory.

First-phase held-out PSNR was:

| Training examples | Seed 0 | Seed 1 | Seed 2 |
|---:|---:|---:|---:|
| 12 | 21.6908 | 21.7910 | 21.7139 |
| 60 | 22.3436 | 22.2960 | 22.3324 |
| 120 | 22.5341 | 22.4907 | 22.4623 |

The largest arms were still improving at the boundary. The authoritative decision therefore uses
the matched low-rate continuation in `../encoder_convergence_v2_continued`, not this incomplete
phase alone. Checkpoints remain on the GPU host; manifests, logs, summaries, and the exact protocol
are retained here.
