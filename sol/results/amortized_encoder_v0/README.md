# Amortized encoder v0.1: the dense-intermediate pivot's first gate

One feed-forward pass (~5M params) encodes a never-seen video window into a canonical
73,728-jewel field. Trained on twelve teacher-generated windows with the fitter's own
stochastic-voxel render loss; slots are video-seeded on a stratified lattice at calibrated
unity coverage (sigma = spacing/2, opacity logit -2.7), so the network starts near
blur-baseline quality and learns residual structure. Step-3000 snapshot audited at native
288x192 over 48 frames while training continues to step 8000.

## Full-resolution one-shot audit (`perceptual_audit_step3000.json`)

| Arm | PSNR | SSIM | LPIPS | Layout PSNR |
|---|---:|---:|---:|---:|
| Fitted jewel ceiling (9,000 opt steps/clip) | 27.52 | 0.9831 | 0.0982 | 30.46 |
| **Encoded one-shot (a single forward pass)** | **23.24** | **0.9422** | **0.4693** | **29.98** |
| Guide upsample baseline | 19.35 | 0.8596 | 0.6762 | 22.61 |
| Best generative mark-space arm (reference) | 14.48 | 0.6031 | 0.6911 | 15.75 |

Per-class encoded PSNR: Basketball 21.53, HorseRiding 23.32, PlayingGuitar 22.42,
ApplyEyeMakeup 25.71. Contact: `encoder_contact_step3000.png` — soft but fully coherent
scenes: court/players/hoop, horse and rider, red guitar with hands, face with makeup hand.

## Reading

1. **First arm ever to beat the blur baseline on every metric** — and by wide margins
   (+3.9 dB PSNR, −0.21 LPIPS, +7.4 dB layout). The macro-layout deficit that defined the
   generative stack's failure is simply gone: 29.98 versus the ceiling's 30.46.
2. **+8.7 dB PSNR / −0.22 LPIPS / +14.2 dB layout over the best mark-space generative arm**,
   at one forward pass versus 20 Euler steps plus topology prediction.
3. Against the predeclared bars (>=25 proven / ~20 refine / <18 redesign) the PSNR verdict is
   "refine": the remaining 4.3 dB to the ceiling is fine texture, visible in the contacts as
   uniform softness plus faint lattice structure. The available levers are exactly the
   fundable ones: training corpus (twelve windows today; the teacher mints more at ~30
   GPU-min/window all-in), trunk capacity, and iterative refinement passes. Sampled-voxel
   held-out PSNR was still rising at the audit snapshot.
4. v0 (no seeded init) scored 12.2 dB — the video-seeded unity-coverage start is worth ~11 dB
   and is the architectural heart of the pivot: encode-as-refinement, not
   reconstruct-from-bottleneck.

## Consequence

The system "text -> teacher video -> one-shot encoder -> persistent editable jewel field" is
now real end to end at watchable-draft quality on held-out content. Next: corpus scale-up via
the teacher (the encoder's data curve replaces the realizer's), refinement iterations toward
the ceiling, and carry-conditioned windows for persistence.
