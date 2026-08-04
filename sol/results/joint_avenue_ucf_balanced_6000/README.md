# Joint Avenue/UCF tokenizer: equal-exposure result

## Question

Can one sparse 45k-jewel tokenizer represent both the fixed-camera Avenue corpus and a differently
shaped UCF basketball fit without losing the scene colors or sacrificing either domain?

## Protocol

- Architecture: grid `32x32x32`, 80 slots/cell, rank encoder, 128 model dimensions, 24 latent
  dimensions, encoder depth 0, decoder depth 4, 8 heads, 8.98M parameters.
- Sampling: batch one, deterministic 50/50 alternation between Avenue and UCF.
- Normalization: train-only moments, with the two domains weighted equally.
- Training: 6,000 total steps, giving each domain 3,000 updates. The 500-step warmup preserves the
  same per-domain warmup exposure as the 250-step single-domain controls.
- Avenue audit: all 35 windows from held-out source videos 03 and 05, 1,024 render points/window.
- UCF audit: the one training basketball window, 4,096 render points. This is a reconstruction and
  shared-capacity diagnostic, not evidence of UCF cross-video generalization.
- All UCF comparisons below use the exact same `ucf_dense_transfer_100m_retime` fitted target.

## Results

| Tokenizer | Avenue held-out PSNR | Avenue count | UCF PSNR | UCF count |
|---|---:|---:|---:|---:|
| Avenue-only, frozen | 19.987 dB | 99.14% | 17.457 dB | 85.83% |
| UCF-only, exact-fit control | — | — | 21.892 dB | 96.80% |
| Joint, 3,000 steps (1,500/domain) | 18.992 dB | 100.61% | 20.288 dB | 93.13% |
| **Joint, 6,000 steps (3,000/domain)** | **20.278 dB** | **100.85%** | **22.275 dB** | **94.04%** |

Equal-exposure joint training beats the Avenue-only result by 0.291 dB and the exact-fit UCF-only
control by 0.383 dB. Evaluating the Avenue-only checkpoint with the joint model's 80-slot decode cap
produces 19.986 dB, so the Avenue gain is not explained by the cap change. The shorter joint run
looked like a capacity tradeoff only because each domain received half as many optimizer updates.

## Visual finding

The frozen Avenue tokenizer shifts the basketball floor and court toward tan/lilac and largely
erases the players. Joint training restores the orange-red court, cooler green floor, and player
locations. Fine people and limbs remain blurred, so global render PSNR is not yet a sufficient gate
for identity-preserving edits.

- `ucf_frame0_three_way.png`: fitted target, frozen Avenue transfer, 6k joint model, UCF-only control.
- `ucf_visuals/v_Basketball_g01_c01_w000000_dense_roundtrip.gif`: complete 16-frame joint result.
- `avenue_visuals/03_w000000_dense_roundtrip.gif` and `05_w000000_dense_roundtrip.gif`: held-out
  Avenue examples.

## Conclusion and next gate

At the 45k target density, the evidence supports a shared tokenizer and does not justify
domain-specific branches. It does **not** validate final visual capacity: contribution-aware audits
show only ~3k effective splats/frame. The next gate is a denser fit before cross-video transfer;
after that, fit a small diverse UCF set, hold out at least one whole video, and repeat equal-exposure
training with foreground/chroma reporting.
