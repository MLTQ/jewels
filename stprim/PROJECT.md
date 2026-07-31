# jewels

## What This Is

A spacetime-primitive representation for video, and the scaffolding to test whether it can
support a *generative* model rather than only per-clip fitting.

A video is treated as a 3D volume in (u, v, t) — not a frame sequence, not keyframe+deltas —
and reconstructed from N anisotropic primitives. Each primitive's orientation tilted into the
t axis encodes velocity, so a moving object is a single sheared tube rather than a new
primitive per frame.

This is NOT a codec. GSVC / VeGaS / GaussianVideo already do per-clip fitting well. The end
goal (restated 2026-07-31) is a **generative video model that emits spacetime primitives** —
generation as splat emission in (u,v,t), not frame-by-frame diffusion. The unbuilt thing is
the amortization step: a shared learned vocabulary across a corpus plus a prior that emits
compositions of it. Stage 1 (here) exists to produce training data for stage 2.

## Current State

Stage 1 fitter works end-to-end on CPU and GPU. Additive splats only — the voronoi branch was
measured, steelmanned, and **removed from the tree** 2026-07-31 (archived at
`jewels/stprim-final-with-voronoi-20260731.tar.gz`; history below). The first GPU run (2026-07-31,
4090 on Aine) required a fix: `PrimitiveField` init passed a CPU generator to CUDA `torch.rand`,
which PyTorch rejects. Draws now happen on the generator's device and move — a seed therefore
gives a bit-identical init on every device.

- `core/`, `models/`, `fit/`, `data/`, `cli/` implemented
- `experiments/canonicalization.py` run at real budget on the 4090 (2026-07-31), synthetic
  tube, 24f/128px, 1500→6000 prims, 2000 steps, seeds 0/1:
  - additive: 55.3 / 54.7 dB, chamfer ratio **0.749**
  - voronoi:  53.9 / 54.7 dB, chamfer ratio **0.825**
  - marginals match within ~3% in both modes → reads as **weakly canonical**: centers do not
    correspond seed-to-seed (ratio nowhere near <<1), but the distribution does. Per the
    experiment's own decision table: permutation-invariant (set-diffusion) prior viable,
    autoregressive prior is not. Contra the docstring's expectation, voronoi was *less*
    canonical than additive — no Lloyd relaxation is implemented yet, so the predicted
    voronoi advantage has no mechanism to show up.
  - Caveat: the synthetic tube is mostly flat background, which under-constrains centers;
    re-run on real textured footage before treating this as the final go/no-go word.
- Real-footage A/B + canonicalization run (2026-07-31; "amplify" clip — fixed-camera stage
  performance, dark background; 64f @ 160×306, 3000→10000 prims, 3000 steps):
  - reconstruction: **additive 43.2 dB vs voronoi 36.8 dB** — the 1.4 dB synthetic gap becomes
    6.4 dB on real content. Voronoi shows cell-boundary artifacts not aligned with image edges
    (the crystalline-mosaic failure, present even with P1); worst on the thin fast-moving mic.
  - canonicality: **additive ratio 0.621, voronoi 0.745**, marginals match closely in both
    modes. Additive is more canonical on real footage too. Voronoi has now lost every measured
    axis, contradicting the docstring's expectation.
  - aniso_med ≈ 2.8–3.3 on real footage (vs ≈1.2 synthetic): fits genuinely use strong
    anisotropy, which supports the core motion-in-orientation premise itself.
  - Standing caveats for the voronoi arm: no Lloyd relaxation implemented (its theoretical
    canonicalization mechanism never ran), and no background model — a partition of unity must
    tile flat regions that additive gets nearly free from its learned global background. The
    cheap steelman would be Lloyd + a background pseudo-cell (constant logit in the softmax).
- Steelman round for voronoi (2026-07-31, same clip/budget; `--steelman` / `--bg-cell` flags):
  - bg pseudo-cell alone: training PSNR ≈ 42.0–43.0 dB — closes the reconstruction gap, so the
    vanilla deficit really was the missing background model, not cells being bad at images.
    Canonicality ratio 0.684: better than vanilla voronoi, still behind additive's 0.621. Note
    the mechanism is additive's background idea ported into the softmax — nothing
    voronoi-specific.
  - + Lloyd (interleaved + full polish): reconstruction collapses to 33.0 dB full-volume (CVT
    equalizes cell sizes; content-adaptive fitting wants the opposite) and *relative*
    canonicality worsens (0.784): absolute cross-seed chamfer improves (0.056 vs 0.070) but the
    random baseline drops faster (0.072 vs 0.094) because Lloyd spreads seeds toward uniform.
    Lloyd canonicalizes local packing, not global assignment — lattice phase and orientation
    remain gauge-free. The docstring's premise fails structurally, not for lack of tuning.
- Stage 2 (amortized encoder / set prior): not started

Smoke result, synthetic tube, 24px/6frames/32 prims/120 steps: additive 29.16 dB, voronoi
23.86 dB on CPU/torch 2.10 (originally logged 26.30/22.77 on an older env); GPU matches.

## Active Work

1. **Corpus generation.** Freeze the checkpoint schema (state/cfg/info as of 2026-07-31),
   collect fixed-camera clips, batch-fit on the 4090 (~3 min/clip at 64f/160px — hundreds per
   night as-is; adopt a GSVC-family CUDA rasterizer only when this becomes the bottleneck).
2. **Primitive featurization for the prior.** Feed the covariance (or its log/Cholesky), not
   (scale, quat): quaternions double-cover rotations (q ≡ -q) and axis order permutes, so raw
   parameters are non-canonical and would poison a prior. Plus color/P1 normalization and a
   position-time encoding.
3. **Prior v0.** Permutation-invariant set generator (set diffusion / flow matching) over
   fitted sets, unconditional, small corpus. Autoregression over the SET is ruled out by the
   canonicality data; autoregression over TIME is not — sorting by t gives the order the set
   lacks, and windowed emission conditioned on overlapping past primitives is the natural
   path to variable-length/streaming generation. Decide after v0 samples exist.

## Architecture Overview

```
data/video_io.py                  decode -> (T,H,W,3) in [0,1]
core/volume.py                    (T,H,W) -> normalized (u,v,t) grid; pixel<->frame metric
core/params.py                    PrimitiveField: mu, log_scale, quat, color[, color_grad], logit_w
models/render.py                  kNN cull -> Mahalanobis -> additive accumulation + learned bg
fit/adapt.py                      gradient-driven densify + weight-driven prune
fit/fitter.py                     stochastic-voxel Adam loop
cli/fit_video.py                  fit one clip -> checkpoint
cli/render_recon.py               fit + visual artifacts (GIF, contact sheet)
experiments/canonicalization.py   two-seed chamfer/marginals measurement
```

## Decision Log

- **[2026-07] Additive splats and soft Voronoi are ONE model, differing only in normalization.**
  logits = -q/2 + log w; additive takes `exp(logits).sum()`, Voronoi takes
  `softmax(logits/tau).sum()`. This is why there is one renderer and not two. It makes the A/B
  a change of argument rather than a comparison of two separately-tuned codebases.
- **[2026-07] Voronoi cell SHAPE cannot be a codebook entry.** A cell's shape is determined by
  neighbouring seeds, not by the seed itself — emit a shape and it contradicts what the
  neighbours imply. The codebook must quantize the per-seed *metric tensor* (anisotropy), which
  IS intrinsic. Original "vocabulary of jewel shapes" framing is dead; "vocabulary of
  normalized anisotropies" replaces it.
- **[2026-07] Anisotropy tensor carries motion.** A metric whose principal axis tilts in t
  produces a sheared tube. Motion does not need a separate parameter or a separate codebook
  axis. This is why quaternion orientation is a free parameter rather than fixed to axis-aligned.
- **[2026-07] P1 (linear color ramp) per primitive, not P0.** Constant-color cells give a
  crystalline mosaic with hard edges at cell boundaries that don't align with image edges, and
  need absurd counts for smooth gradients. Standard FEM reasoning. `--no-p1` to compare.
- **[2026-07] Euclidean kNN culling, not Mahalanobis.** Culling only needs a safe superset;
  computing the true metric against all N defeats the purpose. Far primitives have
  astronomically negative logits either way.
- **[2026-07] Rebuild the optimizer after every adapt step.** Parameter tensors are replaced, so
  Adam moment buffers become invalid. Costs some momentum; the alternative (surgical state
  splicing) is a known source of silent bugs in 3DGS implementations.
- **[2026-07] Stochastic voxel sampling, not whole frames.** Keeps VRAM flat regardless of clip
  length, which is the entire point of the volume framing.
- **[2026-07-31] Voronoi gets one steelman round before keep-or-kill.** Vanilla voronoi lost
  every measured axis, but it was scored without its claimed canonicalization mechanism (Lloyd
  relaxation — never implemented) and with a structural handicap (no background model, so cells
  tile flat regions additive gets free). Built both as default-off knobs (`fit/lloyd.py`,
  `bg_cell` in FitConfig; `--steelman` on the experiment CLIs), additive arm untouched. This is
  deliberately the mode's best case: if it still loses, the branch closes with no lingering
  "but what if". Old results stay reproducible because everything defaults off.
- **[2026-07-31] Voronoi branch closed.** Scoreboard on real footage (full PSNR / chamfer
  ratio): additive **43.2 / 0.621**; voronoi vanilla 36.8 / 0.745; +bg-cell ≈42.5 / 0.684;
  +bg-cell+Lloyd 33.0 / 0.784. Every variant loses both axes; the best variant gets close only
  by borrowing additive's background idea, and the voronoi-*specific* mechanism (Lloyd) hurts
  both axes for a structural reason (lattice-phase gauge freedom). Stage 1 is additive splats —
  the GSVC-family fitting stage. The actual novelty (spacetime volume framing, anisotropy
  vocabulary, set prior) is renderer-agnostic and carries over unchanged. Voronoi code stays
  in-tree behind default-off flags as the record of the test.
- **[2026-07-31] Voronoi code removed from the tree** (supersedes "stays in-tree" above —
  Max's call: a decided branch should not linger). The record is this log, the archived tree
  at `jewels/stprim-final-with-voronoi-20260731.tar.gz`, and the run checkpoints on Aine.

## Sharp Edges

- **`t_scale` is arbitrary.** Nothing in the data fixes the exchange rate between pixels and
  frames. Per-primitive anisotropy can absorb a bad global value, but kNN culling is isotropic
  and *does* care. If culling looks wrong on high-motion footage, this is the first knob.
- **Pure-PyTorch renderer.** No tile rasterizer. Expect it to be 1-2 orders of magnitude slower
  than a CUDA 3DGS kernel. Fine for the go/no-go; the obvious optimization target after.
- **Fixed camera, no cuts.** Global camera motion shears every primitive simultaneously and eats
  the budget before anything is learned about objects. A cut violates volume continuity and
  shows up as a wall of primitives at the boundary.
- **Canonicality is the load-bearing assumption.** If two seeds don't produce comparable
  primitive sets, stage 2 is impossible as designed. Do not build the encoder before checking.
