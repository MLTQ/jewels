# LTX-2.3 scaffold corpus v1

The first prompt-generated scaffold corpus completed on Aine's RTX 4090 on August 11, 2026. It is
a balanced semantic gate for the video-to-jewel realizer: three training phrasings and one held-out
evaluation phrasing for each of Basketball, HorseRiding, PlayingGuitar, and ApplyEyeMakeup.

## Result

| Measurement | Result |
|---|---:|
| Completed / failed | 16 / 0 |
| Geometry | 768x512, 49 frames, 24 fps |
| Duration per clip | 2.041667 s |
| Video / audio | H.264; 48 kHz stereo AAC |
| Aggregate generation time | 2,767.74 s (46.13 min) |
| Mean / min / max per clip | 172.98 / 156.77 / 206.42 s |
| Mean GPU peak | 9,399.31 MiB |
| GPU-peak range | 9,398--9,401 MiB |
| Source-manifest SHA-256 | `5e514966be50240393ff51a65327165e8794edc4898cdecc02269adc982be578` |

All 16 files independently report 768x512 H.264, 49 frames at 24 fps, and 48 kHz stereo AAC.
Receipts and the authoritative generated-corpus manifest remain at
`/home/m/LTX-2/corpora/jewels_ucf_prompt_v1` on Aine.

## Visual audit

Each row below is one prompt/seed sampled at frames 0, 16, 32, and 48. All 16 samples preserve the
requested action and a coherent primary subject over the two-second window. Basketball includes
recognizable ball handling, shooting, and court scenes; HorseRiding preserves horse/rider topology;
PlayingGuitar preserves the instrument and playing pose; ApplyEyeMakeup preserves the close facial
layout, hand, and cosmetic tool. Fine hand/object interactions remain imperfect, as expected from a
general video prior, but the macro-geometry is materially stronger than the washed-out direct jewel
generator.

- [Basketball audit](00_basketball_audit.jpg)
- [HorseRiding audit](01_horseriding_audit.jpg)
- [PlayingGuitar audit](02_playingguitar_audit.jpg)
- [ApplyEyeMakeup audit](03_applyeyemakeup_audit.jpg)

## Throughput interpretation

The observed low average GPU activity is real but not a failed run. Each distilled sample performs
eight half-resolution and three full-resolution denoising steps in about 14 seconds; most of the
roughly 173-second wall time repeatedly streams or rebuilds Gemma, the 22B transformer, upsampler,
and decoders under CPU offload. The official pipeline releases these components inside each call,
so merely retaining a `DistilledPipeline` Python object does not remove the loading cost.

Before scaling by orders of magnitude, benchmark either two concurrent CPU-offloaded workers
(their independent 9.4 GiB peaks should fit in 24 GiB, but shared RAM/PCIe behavior is unmeasured) or
a prompt-embedding/model-lifecycle refactor. The semantic-scaffold gate itself passes.

## Density-matched jewel reconstruction

The first evaluation-prompt scaffold has now been fitted at 72,000 jewels, a budget scaled from the
120k/96-frame contract by spacetime volume. Its exact full-video render reaches 28.182 dB and
measures 6,384 effective contributors/frame on average (5,485–7,626), directly satisfying the
5k–10k density target. Court/action/color geometry survives, while fine structures and fast limbs
remain softer than the LTX source. See [`jewel_fit_gate_72k/README.md`](jewel_fit_gate_72k/README.md)
for the comparison and density report.

The second, Horse Riding, fit reaches 31.707 dB and 7,134 effective contributors/frame while
preserving the rider, horse, arena geometry, and blue shirt. This makes the density result a
two-content confirmation rather than a Basketball-specific calibration.

Playing Guitar reaches 30.839 dB and retains its saturated red instrument and moving hands. Its
5%-alpha contributor count is 5,339/frame, while opacity-weighted effective density is 4,597/frame;
the strong visual despite that stricter shortfall motivates content-aware density control rather
than blind top-up to a fixed scalar count.

Apply Eye Makeup completes the frozen four-class reconstruction gate at 35.158 dB, with 5,749
effective contributors/frame and 6,762/frame above 5% peak alpha. The face, hand, eye motion, and
palette remain stable. Across all four 72k fits, mean replay PSNR is 31.472 dB and mean effective
density is 5,966 contributors/frame.

## Cross-domain jewel realizer

The UCF-trained raster-guided stochastic mark flow passes its predeclared evaluation on the four
unseen LTX scaffolds. Guided projected samples reach 15.342 dB / 0.6942 SSIM versus 13.777 dB /
0.4160 for deterministic continuation; target-relative contrast moves from 0.426 to 0.743, edge
energy from 0.486 to 0.830, and birth-cell adherence reaches 99.08%. HorseRiding, PlayingGuitar,
and ApplyEyeMakeup retain recognizable primary subjects/actions; Basketball remains visibly the
weakest transfer.

This licenses the hybrid video-scaffold-to-jewel path only. Correct and shuffled text are tied once
the LTX future raster is provided, and target topology remains privileged. See
[`realizer_eval/README.md`](realizer_eval/README.md) for the frozen split, full metrics, and videos.

## Learned topology and generated marks

The target-topology dependency is now removed for one 16-frame continuation stride. A 0.88M
cell-RGB/carry topology head trained on the 12 UCF fields reaches 0.6636 held-out LTX slot F1 versus
0.6092 shuffled, 0.6091 null, and 0.6222 for a per-stride train mean. When every predicted rank is
synthesized by the frozen realizer, learned topology reaches 15.464 dB / 0.6967 SSIM, essentially
matching the oracle-topology 15.492 dB / 0.6997 result. It averages 5,822 effective
contributors/frame across all four classes, predicts 99.68% of the target birth budget, and copies
carried features with exactly 0.0 error.

The learned/shuffled rendering gap is modest because topology cells are nearly dense and the mark
flow still receives the correct LTX guide, but count correlation and slot F1 establish causal
scaffold use. Visual noise is shared by learned and oracle topology, localizing the remaining
quality problem in mark realization. See [`topology_eval/README.md`](topology_eval/README.md) for
the full controls, GIFs, cross-validation diagnostic, and checkpoint hashes.

## Autonomous rollout and appearance factorization

The fitted initial field and fitted continuation carry have now been removed for all three 16-frame
windows. A corrected left-censored clip-start boundary raises the frozen generator to 14.570 dB /
0.6306 SSIM with 7,493 effective contributors/frame while retaining append-only stable IDs and
bit-exact carried rows. Correct scaffolds beat shuffled by 4.116 dB. See
[`scaffold_mark_rollout/README.md`](scaffold_mark_rollout/README.md) for the three-window controls.

A two-stream architectural control then freezes topology, density, lifecycle, and IDs while allowing
only an RGB residual in the top 20% scaffold-salient cells. The selected full-flow control gains
0.034 dB, improves foreground/detail metrics in three classes, and lowers macro quiet error across
12 correct/shuffled/null rollouts. See
[`lifecycle_appearance_ablation/README.md`](lifecycle_appearance_ablation/README.md).

The purpose-built 74,067-parameter adapter preserves the same exact ownership with 28.7 times fewer
trainable parameters. At 288x192, teacher distillation plus half-strength calibration improves
PSNR and foreground PSNR in all four classes (+0.00641 dB and +0.02935 dB macro), lowers macro
edge/motion/quiet errors, and keeps every class inside the quiet gate. It nevertheless captures
only 29% of the full-flow PSNR gain under the exact matched protocol; full strength captures 57%
and fails the native class-level quiet gate. It is retained as a structural proof, not selected as
a teacher replacement. Native-aspect contacts also confirm that RGB correction is much smaller
than the remaining coherent mark-realization problem. See
[`appearance_adapter_ablation/README.md`](appearance_adapter_ablation/README.md).

## Neighborhood-coupled birth sets

The first coupled-set realizer adds one 414,016-parameter (+19.48%) zero-residual block to the
frozen v1 flow. It pools learned jewel hidden states by cell, mixes them across the 3D neighborhood,
and broadcasts shared composition state without changing topology, ranks, carry, IDs, or renderer.
The selected step-2,250 screen lowers held-out fixed-path error by 0.221%. On the exact old 40x24
protocol it improves PSNR by 0.405 dB, foreground PSNR by 0.877 dB, and lowers macro edge, motion,
and quiet errors, though SSIM drops 0.00444. The primary 288x192 autonomous gate is stronger:
macro PSNR/SSIM improve by 0.334 dB/0.00300, foreground PSNR rises 0.658 dB, all four classes
improve PSNR, foreground fidelity, edge error, and quiet stability, and three improve SSIM and
motion-boundary error. Visual gains remain subtle beside the large gap to fitted fields. An exact
base-owned-count audit retains +0.332 dB PSNR, +0.761 dB foreground PSNR, and all-class
edge/motion/quiet gains, but SSIM improves in only two classes and the visual-recognition gate
fails. The coupling direction is retained; the feature-loss checkpoint is not selected. See
[`coupled_set_v1/README.md`](coupled_set_v1/README.md) for the native-aspect gate and decision.

## Guide-upsample baseline

The rollout's own input, decoded trivially, now bounds every pixel-fidelity claim. Trilinear
upsampling of the exact per-stride `(16,16,8)` cell-RGB guides reaches 21.081 dB / 0.9037 SSIM on
the four held-out scaffolds — beating the generated-correct rollout (14.570 / 0.6306) in every
class and sitting only 0.76 dB below the fitted 72k ceiling — while carrying only half the
target's edge energy (0.515) and temporal change (0.508). The generated field restores both to
~1.2 at target-level density. Single-reference PSNR/SSIM therefore cannot demonstrate the
generative stack's contribution; detail/motion-energy restoration and the editable persistent
field must carry the claim, and perceptual/distribution metrics are required. See
[`guide_upsample_baseline/README.md`](guide_upsample_baseline/README.md).
