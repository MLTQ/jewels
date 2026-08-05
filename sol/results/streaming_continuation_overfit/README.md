# Prefix-conditioned continuation overfit

> Superseded by the matched cell-local model in
> [`../streaming_continuation_local/README.md`](../streaming_continuation_local/README.md), which
> improves correct-context field PSNR by 3.41 dB.

## Outcome

The first learned streaming gate passes on one jointly fitted 96-frame UCF basketball field.
Given a 32-frame prefix, a 0.80M-parameter model predicts the variable-count jewel births for the
next 16 frames while persistent jewels are carried by stable ID without entering the model.

The four continuation views contain 13,722–19,296 births and 18,601–19,987 carried jewels. The
`16×16×8` birth grid needs at most 147 of its 256 slots in any cell. The 5,000-step CPU overfit took
569.1 seconds.

## Final control gate

The saved checkpoint was evaluated at 16 sampled render points per future frame. Shuffled prefixes
are chosen only when their source intervals are disjoint from the evaluated target stride.

| Metric | Correct prefix | Shuffled prefix | Null prefix |
|---|---:|---:|---:|
| Standardized birth-feature MSE | **0.6946** | 0.8531 | 0.8726 |
| Birth count MAE per cell | **1.2130** | 3.4591 | 5.4854 |
| Sampled future-field PSNR | **16.459 dB** | 15.641 dB | 15.847 dB |

Correct context reduces feature error by 18.6% versus shuffled and 20.4% versus null, and improves
render PSNR by 0.82 dB and 0.61 dB respectively. Independently decoded births recover 99.46% of the
target count. Carried jewels have 0.0 maximum error.

## Interpretation and limits

This proves that the learned birth process uses the preceding jewel field; it is not merely an
unconditional average continuation. It also validates the crucial state split: long-lived jewels
are persistent state, while only births consume generative capacity.

This remains an overfit, not a generation-quality claim. Feature/render evaluation uses oracle
target cell ranks and counts to isolate mark prediction; the variable-count path is reported
separately. Generalization, free-running multi-window rollout, prompt conditioning, and pixel-level
visual quality are still open gates.

The intermediate records in `train_log.jsonl` used an earlier adjacent-window shuffled control
that overlapped the target interval. Those shuffled render numbers contain future leakage and are
not authoritative. `summary.json` contains the corrected, denser final evaluation.

## Reproduction

The source is the 120k spatial-split checkpoint
`ucf_streaming_96f_120k_spatial/v_Basketball_g01_c01_w000000.pt`, fitted jointly over 96 frames at
33.254 dB. The training command was:

```bash
python -m sol.train_streaming_continuation \
  --checkpoint /private/tmp/ucf_96f_120k_spatial.pt \
  --out sol/results/streaming_continuation_overfit \
  --device cpu --steps 5000 --warmup 200 \
  --model-dim 64 --context-depth 2 --cell-depth 2 --slot-depth 2 \
  --eval-every 1000 --eval-points 4 \
  --checkpoint-every 500 --log-every 50 --no-amp
```
