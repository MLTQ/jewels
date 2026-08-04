# Hierarchical axial generation and editing spike

## Decision

The representation and editor mechanics are feasible; the current data/model combination is not
yet a useful text-conditioned generator or visual inpainter.

- A fixed 2³ PCA hierarchy reduces the successful 32³ tokenizer to 4,096 coarse codes while losing
  only 0.224 dB of held-out sampled-render PSNR.
- A 1.94M-parameter axial flow runs comfortably on the 8 GB RTX 2070 SUPER and learns the held-out
  latent distribution better than retrieval and scene-mean collapse.
- It does not use the current CLIP condition: correct conditioning is slightly worse than shuffled
  or unconditional conditioning.
- Real jewel translation, conservative dirty-block mapping, protected-jewel merge, and exact clean
  clamping all work. The learned dirty region remains visibly washed out.

This is a strong architectural proof and a clear data/learning stop signal, not an end-quality pass.

## Hierarchy result

The selected block codec groups each 2³ neighborhood of 24-D fine latents and retains 96 of 192
PCA dimensions:

| Measurement | Result |
|---|---:|
| Fine raster | 32³ × 24 |
| Coarse raster | 16³ × 96 |
| Token-count reduction | 8× |
| Numeric reduction | 2× |
| Train-only explained variance | 94.749% |
| Held-out normalized fine-latent MSE | 0.13650 |
| Full 35-window render PSNR | 19.763 dB |
| Loss versus fine tokenizer | 0.224 dB |

The matched source-`03` and source-`05` artifacts are
`block_pca96_03_w000000_roundtrip.gif` and `block_pca96_05_w000000_roundtrip.gif` in the sibling
`dense/` directory. Pedestrian silhouettes survive similarly to the fine tokenizer, so the
hierarchy passes its visual gate.

## Full-generation prior

- Grid: 16³, 96-D codes
- Architecture: six rotating u/v/t axial-attention blocks, model width 128
- Parameters: 1.94M
- Training: 10,000 total steps, batch 2, fp16
- Evaluation: 35 windows from unseen videos `03` and `05`; 16 generated samples, 25 Euler steps,
  CFG 1.0

| Measurement | Result |
|---|---:|
| Sample energy distance | **0.2493** |
| CLIP-retrieval energy distance | 0.3164 |
| Scene-mean energy distance | 0.8473 |
| Conditional flow MSE | 3.92698 |
| Shuffled-condition flow MSE | 3.92675 |
| Unconditional flow MSE | 3.92527 |
| Conditional gain | **-0.00171** |

Lower energy distance is better. `03_w000000_axial_comparison.gif` and
`05_w000000_axial_comparison.gif` show that samples reproduce the camera-scene family and temporal
texture but not prompt-specific content. With only six camera sources total (four train, two held
out) and one image-CLIP sidecar per window, this is principally a condition-diversity failure.

## End-to-end moved-region edit

`03_w000000_maskaware_edit.gif` is the controlled editor artifact. It translates 281 fitted jewels
by `-0.3` in normalized horizontal space, marks the swept selection plus halo, repairs touched
hierarchy blocks, and merges moved jewels as protected constraints.

| Measurement | Result |
|---|---:|
| Fine dirty cells | 2,548 / 32,768 |
| Coarse dirty codes | 448 / 4,096 (10.94%) |
| Fine cells affected by block decode | 3,584 |
| Clean coarse max error | **0.0** |
| Clean fine max error | **0.0** |
| Protected moved jewels | 281 |
| Final merged jewels | 45,244 |

The untouched 89.06% of coarse codes and their decoded fine blocks are exact. The video therefore
stays sharp outside the repair strip; the strip itself is blurry.

## Repair training controls

Two 5,000-step branches used identical held-out cuboids (8 examples, mean 5.77% dirty):

| Branch | Base dirty MSE | Final dirty MSE | Interpretation |
|---|---:|---:|---|
| Mask hidden from model | 1.8894 | 1.8926 | No learning; request is ambiguous |
| Learned clean/dirty embedding | 1.8894 | 1.8848 | Marginal; no visible improvement |
| Normalized zero fill | — | 1.0847 | MSE-favored conditional-mean control, not a sample-quality metric |

The mask-aware branch briefly reached 1.8681 at step 2,000, then regressed. Both final renders are
nearly indistinguishable. This rules out further small hyperparameter sweeps on the same 86 training
windows as the best next use of compute.

## Recommended next experiment

1. Expand from six fixed-camera videos to a prompt-captioned, scene-diverse corpus. Conditioning
   cannot be proved when prompts are nearly confounded with four training cameras.
2. Train on edit tuples, not only clean windows: random jewel transforms, explicit protected-jewel
   summaries, dirty masks, and original targets.
3. Add a learned local residual decoder above the 96-D PCA code. The current fixed block inverse is
   excellent for hierarchy validation but forces one coarse sample to determine all eight fine
   cells linearly.
4. Evaluate repair with rendered boundary/perceptual metrics and multiple samples; retain exact
   clean-cell checks as a hard invariant. Per-sample latent MSE alone rewards mean filling.

## Artifacts

- `eval_cfg1_16.json`: full held-out generation protocol
- `summary.json`, `train_log.jsonl`: full-generation run
- `maskblind_summary.json`, `maskblind_train_log.jsonl`: failed implicit-mask repair branch
- `maskaware_summary.json`, `maskaware_train_log.jsonl`: explicit-mask repair branch
- `edit_manifest.json`, `maskaware_edit_manifest.json`: edit locality audits
- `03_w000000_hierarchical_edit.gif`: base-prior edit
- `03_w000000_maskaware_edit.gif`: mask-aware-prior edit
