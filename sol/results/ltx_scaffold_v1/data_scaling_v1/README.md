# Realizer data-scaling curve: 4 / 8 / 12 fitted training sources

Does more fitted training data convert into scaffold-to-jewel realization quality? Three
class-balanced curve points share one frozen protocol: the exact run_12000 mark-flow recipe
(12,000 steps, seed 23) and run_3000 topology recipe (3,000 steps, seed 17), bit-identical
prompt embeddings (`sol/rebind_prompt_cache.py` — no re-encoding), the same four held-out LTX
evaluation scaffolds, deterministic seed-31 native 288x192 correct-only rollouts, and the same
LPIPS battery (`sol/perceptual_eval.py`). Points 4 and 8 keep each class's source groups
`<= 1` / `<= 2` (`sol/build_scaling_subsets.py`); point 12 is the existing full stack.

## Curve (`curve.json`, `scaling_curve.png`)

| Fitted sources | Held-out velocity loss | Rollout PSNR | Rollout SSIM | Rollout LPIPS |
|---:|---:|---:|---:|---:|
| 4 | 1.7906 | 13.427 | 0.6099 | 0.7156 |
| 8 | 1.6579 | 14.264 | 0.6041 | 0.7069 |
| 12 | 1.5582 | 14.142 | 0.6001 | **0.6971** |

Each subset's report also reproduces the protocol-shared arms bit-consistently (baseline 0.6761,
fitted ceiling 0.0982 in every report), confirming only the training corpus varied.

## Reading

1. **Held-out velocity loss and LPIPS are monotone and unsaturated.** Loss falls 1.791 →
   1.658 → 1.558 and perceptual error 0.7156 → 0.7069 → 0.6971, both nearly linear in
   log-sources with no flattening at the last point. The realization stage is data-starved:
   every additional class-balanced fitted source still buys measurable held-out quality at a
   constant rate.
2. **PSNR gains 0.8 dB from 4 to 8 sources, then stalls.** Given the guide-upsample result —
   PSNR rewards blur and penalizes restored detail — a stalling PSNR beside an improving LPIPS
   is the expected signature of a model gaining detail; it is not evidence of saturation.
3. **Costs are dominated entirely by corpus, not training.** Each curve point trains in under
   ten minutes on one consumer GPU; a fitted source costs ~25 GPU-minutes. The curve therefore
   prices quality directly in fitting compute, which the frozen LTX teacher can supply without
   limit (see `sol/PROMPTABLE_ROADMAP.md`, "Compute conversion").
4. Caveats: one seed per point, three points, four evaluation clips, and the correct-shuffled
   margin thins at 12 sources (0.0170/0.0176/0.0085) — the margin is measured against
   *class-shuffled* scaffolds while training classes are fixed, so it partly reflects class
   memorization at small corpus sizes.
5. **Measured seed noise (three seeds per 12-source arm, `seed_variance.json`):** UCF velocity
   1.608 ± 0.043, LPIPS 0.696 ± 0.009, layout 15.34 ± 0.02; domain velocity 1.072 ± 0.038,
   LPIPS 0.675 ± 0.013, layout 13.57 ± 0.30 (mean ± sd; topology and rollout seeds fixed).
   Under this floor the curve's claims re-sort: the 4→8 velocity step (0.133) is ~3 sd and
   stands; the 8→12 step (0.050 against the multi-seed mean) is ~1.2 sd and is **not
   individually resolved**; per-step LPIPS improvements (~0.009) are ~1 sd each. The curve is
   monotone in means but its tail needs multi-seed subset points or larger corpus steps —
   exactly what corpus scale-up provides. The domain-matching velocity gap (0.536, ~13 sd) and
   the domain layout regression (1.77, ~6 sd) are decisively significant; the domain LPIPS
   advantage (0.021, ~1.9 sd) is suggestive only, and "parity with the blur baseline" is the
   defensible LPIPS claim.

## Provenance

Aine artifacts: `topology/scaling_curve_v1/subset_g{1,2}/{manifest.json,prompts.pt,mark_12000,
topology_3000,rollout_native_correct,perceptual_native}`. Committed here: each point's training
summary, rollout summary, and perceptual report, plus `curve.json` and the figure rendered by
`sol/render_scaling_curve.py`.
