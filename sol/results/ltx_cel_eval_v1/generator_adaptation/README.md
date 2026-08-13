# Cel-field scaffold-generator adaptation

This is the user-directed style-adaptation gate on the four completed 72k cel/rotoscoped jewel
fields. It deliberately proceeds before the fixed-budget fill/contour allocation study selected by
the reconstruction gate, so it tests whether the existing generator can learn this small styled
domain; it does not overturn the contour-allocation diagnosis.

## Corpus and validation contract

The physical corpus contains one 49-frame LTX field for each of Basketball, HorseRiding,
PlayingGuitar, and ApplyEyeMakeup. Each field supplies an initial 16-frame view and two 16-frame
continuation views. The largest observed rank is 341 in a 1,024-rank-per-cell model, so no target is
clipped by capacity.

There is only one physical cel field per class. The manifest therefore gives every field one unique
training alias and one unique reconstruction-audit alias while both aliases resolve to the same
video and exact fit. It records `validation_is_unseen=false`, `source_overlap=true`, paired source
IDs, and `same-field-training-reconstruction`. Metrics in this folder measure adaptation or
memorization, never unseen-field or unseen-prompt generalization.

The adapted manifest contains no photoreal fitted target. It preserves the frozen OpenCLIP prompt
rows from the original four-class corpus under a new manifest digest. LTX receipts own the exact
serialized source-manifest SHA-256, while the prompt cache owns canonical JSON; both are retained.

## Training

Both models use a fresh optimizer and scaler after model-only transfer from the selected UCF
checkpoints. Architecture, model arguments, `16x16x8` grid, 1,024 ranks per cell, and source digest
are validated before loading; destination checkpoints serialize both manifest digests and the
source step.

| Model | Parameters | Source step | Adaptation | GPU | Runtime |
|---|---:|---:|---:|---|---:|
| Stochastic scaffold mark flow | 2,125,846 | 12,000 | 3,000 steps, LR 1e-4 | RTX 4090 | 40.56 s |
| Scaffold topology head | 881,282 | 3,000 | 1,500 steps, LR 1e-4 | RTX 2070 SUPER | 22.90 s |

The mark schedule contains 12 distinct views and repeats the four initial views once, for 16 cyclic
schedule rows. Correct-scaffold normalized feature loss is 0.9732, versus 0.9991 shuffled and
0.9853 null. The separation exists in both regimes: shuffled-minus-correct is 0.0338 initially and
0.0219 in continuation. This is a conditioning-sensitivity result, not a pixel metric. One control
in `mark_summary.json` deserves explicit mention: the no-context arm scores 0.9768 aggregate —
only 0.0036 above correct, and exactly 0.0 above it in the initial regime — so the causal
carried-state context contributes little to mark loss on these same-field targets; the strong
separations above are carried by the scaffold guide, not by context.

The topology reconstruction audit predicts 274,861 births versus 275,251 targets (ratio 0.99858),
with 0.1284 cell-count MAE, 0.9972 count correlation, and 0.99426 slot F1. Shuffled topology falls
to 0.67805 slot F1 and null topology to 0.67534, confirming that the head reads the styled raster
rather than merely replaying an aggregate count prior.

Three-stride oracle-mark topology rollouts retain 98.84% of target effective density on average,
produce 5,895 effective contributors per frame, preserve unique stable IDs, and keep carried
features exact. Per-class effective density is 7,523 Basketball, 4,222 HorseRiding, 6,078
PlayingGuitar, and 5,757 ApplyEyeMakeup.

## Fitted-seed-free visual rollout

The adapted topology and mark flow were coupled for a deterministic, 20-step, seed-31 rollout on
the RTX 4090. No fitted mark, count, carry, or background enters generation. Each control begins at
frame zero and feeds its own generated state through two continuation frontiers, producing 48
frames and an editable field with exact stable IDs.

| Macro over four fields | PSNR | SSIM | Effective jewels/frame | Quiet temporal MAE |
|---|---:|---:|---:|---:|
| Fitted jewel ceiling | 23.131 dB | 0.8974 | 5,964 | 0.00574 |
| **Generated correct scaffold** | **15.401 dB** | **0.4270** | 6,093 | 0.02973 |
| Generated shuffled scaffold | 12.642 dB | 0.0571 | 5,988 | 0.03768 |
| Generated null scaffold | 14.758 dB | 0.0741 | 4,611 | 0.02554 |

Correct conditioning wins SSIM in every class and beats shuffled by 2.759 dB / 0.3699 SSIM. It
also beats null by 0.643 dB / 0.3529 SSIM. The contact sheet makes the learned structure visible:
the correct branch recovers the horse/rider silhouette, red guitarist and instrument axis, face
layout, and basketball-court organization; shuffled and null controls mostly collapse into
class-inappropriate texture.

The result is not yet visually clean. Correct quiet-region temporal error is 5.18x the fitted
ceiling and motion-boundary error is 2.29x. Correct temporal-change energy is 6.41x the source, so
the sparkle/noise around motion remains the principal failure. This is not a low-density failure:
correct generation has 2.15% **more** effective contributors than the fitted ceiling. Free-running
topology also falls from the same-field oracle-carry audit's 0.994 slot F1 to 0.833, showing that
generated carry drift compounds the mark-appearance error even though total count remains 0.99985
of target.

## Interpretation and next gate

This proves the existing topology/mark stack can adapt to all four dense cel fields, generate a
recognizable action/layout from frame zero, and retain a strong scaffold-control signal through
free-running continuation. It also localizes the next problem: not jewel count, but mark appearance
and temporal stability under generated carry. The compact scaffold-gated appearance adapter and
the fixed-budget fill/contour allocator are therefore complementary next gates, followed by a
physically disjoint styled-field evaluation (`jewels-9ld`).

This remains a text-plus-video-scaffold generator. A fully native prompt-only jewel model still
needs either the upstream prompt-to-scaffold stage or a learned replacement for its video guide.

## Reproduction

Commands ran from `/home/m/jewels` with
`PYTHONPATH=/home/m/jewels:/home/m/.cache/uv/archive-v0/1Dhhq76yZjDB8ZZDuQEKU` and
`/home/m/.venv-diffusion/bin/python`. GPU UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`
is the RTX 4090; `GPU-4e207c93-ed93-c35e-f0f2-e37c8df2b047` is the RTX 2070 SUPER.

```bash
python -m sol.build_ltx_style_train \
  --ucf-manifest /home/m/jewels/sol/results/prompt_smoke/manifest.json \
  --ltx-manifest /home/m/jewels/sol/results/ltx_cel_eval_v1/manifest.json \
  --source-prompt-cache /home/m/jewels/corpus/ucf_prompt_smoke/prompts.pt \
  --out-manifest /home/m/jewels/topology/ltx_cel_generator_v1/manifest.json \
  --out-prompt-cache /home/m/jewels/topology/ltx_cel_generator_v1/prompts.pt

CUDA_VISIBLE_DEVICES=GPU-21d45575-7ece-a97c-35a0-294f7bce9c39 python \
  -m sol.train_scaffold_mark_flow \
  --manifest /home/m/jewels/topology/ltx_cel_generator_v1/manifest.json \
  --prompt-cache /home/m/jewels/topology/ltx_cel_generator_v1/prompts.pt \
  --checkpoint-root /home/m/jewels/corpus/ltx_cel_eval_v1_72k \
  --out /home/m/jewels/topology/ltx_cel_generator_v1/mark_transfer_3000 \
  --device cuda:0 --steps 3000 --lr 0.0001 --warmup 100 --initial-repeat 2 \
  --eval-every 500 --checkpoint-every 250 --log-every 25 --seed 31 \
  --transfer-from /home/m/jewels/topology/scaffold_mark_flow_v1/run_12000/scaffold_mark_flow.pt

CUDA_VISIBLE_DEVICES=GPU-4e207c93-ed93-c35e-f0f2-e37c8df2b047 python \
  -m sol.train_scaffold_topology \
  --manifest /home/m/jewels/topology/ltx_cel_generator_v1/manifest.json \
  --prompt-cache /home/m/jewels/topology/ltx_cel_generator_v1/prompts.pt \
  --checkpoint-root /home/m/jewels/corpus/ltx_cel_eval_v1_72k \
  --out /home/m/jewels/topology/ltx_cel_generator_v1/topology_transfer_1500 \
  --device cuda:0 --steps 1500 --batch-size 8 --lr 0.0001 --warmup 100 \
  --eval-every 250 --checkpoint-every 250 --log-every 25 --seed 31 \
  --transfer-from /home/m/jewels/topology/scaffold_topology_v1/run_3000/scaffold_topology.pt

CUDA_VISIBLE_DEVICES=GPU-21d45575-7ece-a97c-35a0-294f7bce9c39 python \
  -m sol.render_scaffold_mark_rollout \
  --topology /home/m/jewels/topology/ltx_cel_generator_v1/topology_transfer_1500/scaffold_topology.pt \
  --mark-flow /home/m/jewels/topology/ltx_cel_generator_v1/mark_transfer_3000/scaffold_mark_flow.pt \
  --manifest /home/m/jewels/topology/ltx_cel_generator_v1/manifest.json \
  --prompt-cache /home/m/jewels/topology/ltx_cel_generator_v1/prompts.pt \
  --checkpoint-root /home/m/jewels/corpus/ltx_cel_eval_v1_72k \
  --out /home/m/jewels/topology/ltx_cel_generator_v1/rollout_seed31_20 \
  --device cuda:0 --steps 20 --seed 31 --height 48 --width 80 --upscale 4 --deterministic
```

## Artifacts and remote recovery

- `manifest.json` / `prompts.pt`: exact adaptation ownership and frozen prompt cache.
- `run_summary.json`: compact training, control, density, and transfer record.
- `mark_summary.json` / `topology_summary.json`: complete trainer summaries.
- `mark_train_log.jsonl` / `topology_train_log.jsonl`: recovery/evaluation histories.
- `rollout_summary.json`: complete end-to-end visual, saliency, seam, density, topology, and
  provenance report.
- `three_window_rollout_contact.png`: source, fitted ceiling, correct, shuffled, and null panels
  for all four classes; the four GIFs retain every generated frame.
- Aine checkpoints: `/home/m/jewels/topology/ltx_cel_generator_v1/mark_transfer_3000` and
  `/home/m/jewels/topology/ltx_cel_generator_v1/topology_transfer_1500`.
- Deterministic visual rollout: `/home/m/jewels/topology/ltx_cel_generator_v1/rollout_seed31_20`.

## Follow-up single-field capacity gate

The subsequent exact-topology/fitted-carry PlayingGuitar memorization audit is
tracked in `sol/results/ltx_cel_single_guitar_v1`. It rejects strong whole-mark
render/frontier tuning: source PSNR falls from 17.426 to 13.723 dB while effective
density inflates from 6,258 to 7,933 splats/frame. A 100x weaker rendered loss
preserves density and modestly improves spatial SSIM/foreground error, but its
matched endpoint slightly worsens quiet temporal error. The selected next gate is
therefore the frozen-lifecycle, appearance-only adapter (`jewels-brv`), not more
splats or immediate corpus expansion.
