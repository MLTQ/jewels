# UCF-trained realizer on LTX validation scaffolds

This folder holds the leakage-safe cross-domain evaluation contract for the first prompt-generated
scaffold test.

- Training remains the exact 12 UCF source-group clips used by the oracle-guide model.
- Validation replaces only the four UCF group-4 rows with the same-class LTX evaluation-prompt
  clips (49 frames each).
- The 16 frozen 512-D prompt vectors are byte-identical to the training sidecar, but explicit
  example ownership and the manifest digest are rebuilt for the LTX validation source IDs.
- Train-only jewel feature standardizers therefore remain unchanged. LTX videos affect validation
  only.

`manifest.json` contains 12 training and four validation examples. `prompts.pt` matches its digest,
contains finite unit-normalized embeddings (maximum norm error `1.2e-7`), and records all four LTX
validation IDs. All four 49-frame/72k fitted fields completed before the frozen realizer was
evaluated on the first 16-frame continuation view from each validation scaffold.

## Predeclared transfer gate

Before viewing the four-class output, the selected v1 guided flow must beat the deterministic
UCF continuation baseline in macro PSNR and SSIM, move both target-relative contrast and edge
energy toward 1.0, retain at least 98% mean birth-cell adherence after projection, and preserve a
recognizable primary subject/action in at least three of four contact sheets. The true future and
target topology remain privileged, so passing licenses cross-domain mark realization only; it does
not establish free text-to-jewel generation.

## Result: the privileged cross-domain realizer gate passes

| Panel | PSNR | SSIM | Contrast | Edge | Saturation | Temporal | Birth-cell adherence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic continuation | 13.777 | 0.4160 | 0.4261 | 0.4860 | 0.9026 | 0.7873 | 10.21% |
| Projected flow, no guide | 12.810 | 0.3454 | 0.5048 | 0.6691 | 0.8181 | 0.9583 | 97.39% |
| **Projected flow, true LTX guide** | **15.342** | **0.6942** | **0.7428** | **0.8300** | **0.9533** | **0.9815** | **99.08%** |
| Guided flow, shuffled text | 15.319 | 0.6955 | 0.7429 | 0.8260 | 0.9526 | 0.9853 | 99.04% |

The selected UCF-trained realizer beats deterministic continuation by 1.565 dB and 0.2782 SSIM.
Contrast and edge energy both move materially toward the target ratio of 1.0, and projected
birth-cell adherence clears the 98% threshold. Every class improves in PSNR and SSIM; the per-class
PSNR gains range from +0.590 dB for HorseRiding to +2.627 dB for ApplyEyeMakeup.

Visual review passes the declared three-of-four requirement. HorseRiding retains the horse/rider,
PlayingGuitar retains the red guitar and playing pose, and ApplyEyeMakeup retains the face, hand,
and eye-level action. Basketball retains the court and player layout but is the weakest and
blurriest transfer, so it is not counted as a visual pass.

Correct and shuffled text remain effectively tied once the true LTX future raster is supplied:
correct text is +0.023 dB in PSNR but -0.00134 in SSIM. The result therefore validates a
cross-domain **video-scaffold-to-jewel realizer**, not independent prompt understanding by the
jewel model. Target birth cells/ranks and the future video guide are still privileged. The next
causal gate is scaffold-conditioned topology and density generation under overlap carry.

## Artifacts

- `transfer_gate_summary.json`: predeclared gate outcome, macro metrics, and per-class deltas.
- `transfer_gate_contact.png`: one matched target/deterministic/guided frame for all four classes.
- `visual_contract_projection/mark_flow_visual_report.json`: authoritative full-precision report.
- `visual_contract_projection/*_controls.gif`: all 16 evaluated future frames per class.
- `visual_contract_projection/*_contact.png`: three-time, eight-panel controls per class.
