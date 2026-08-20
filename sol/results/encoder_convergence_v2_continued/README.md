# Support-correct encoder scaling decision

## Result

**The amortized video-to-splat bottleneck scales on held-out data.** After equalized corpus-pass
training and a common low-rate continuation, mean support-rendered validation PSNR rises at both
budget increases:

| Examples | Mean PSNR | 95% CI | LPIPS | Layout PSNR | Median mixed tilt |
|---:|---:|---:|---:|---:|---:|
| 12 | 21.876 dB | [21.751, 22.000] | 0.4728 | 29.812 dB | 0.0479 |
| 60 | 22.399 dB | [22.344, 22.453] | 0.4651 | 30.726 dB | 0.0725 |
| 120 | 22.628 dB | [22.459, 22.797] | 0.3916 | 31.533 dB | 0.1875 |

Every 120-example seed beats every 60-example seed. LPIPS falls 15.8% from 60 to 120 examples, and
mixed space/time tilt grows 2.58× rather than collapsing toward axis-aligned lattice dots.

## Protocol

- Exact nested 12/60/120-example subsets from five visual styles and 12 actions.
- Three independent model/training seeds at every budget.
- Frozen 60-example validation inventory, SHA-256
  `3e034a1263e21b566c79b63a8257e9794c4c66316e3a82d6fe65d418d1699b25`.
- Support-complete five-sigma tiled rendering with fatal capacity overflow.
- First phase: 120 corpus passes; continuation: up to 80 further passes at 10× lower learning rate.
- Continuation starts from the matched 120-pass checkpoint rather than selecting different exposure
  counts for different arms.

## Prompt-retention control

The largest encoder was also tested on 12 held-out action prompts through the conservative
prompt → pretrained-video scaffold → encoder → renderer route. Mean rendered CLIP similarity is
0.3339 with the correct prompt, 0.1427 with shuffled prompts, and 0.2096 with null text. Correct
beats shuffled in 12/12 cases and retains 91.4% of the source video's prompt alignment.

This is evidence that encoding and support rendering preserve semantics. It is not evidence for a
direct prompt-to-splat prior, which remains the next gate.

## Artifacts

- `aggregate/convergence.png` and `aggregate/report.json`: replicated learning curve.
- `audit/perceptual_structure.png` and `audit/audit.json`: full-frame perceptual/layout/geometry
  audit across styles and seeds.
- `audit/qualitative.png`: five-style source and 12/60/120-example render comparison.
- `prompt_smoke/prompt_controls.png` and `prompt_smoke/report.json`: correct/shuffled/null text
  controls.
- `n*/seed*/`: compact run summaries and training logs. GPU checkpoints are intentionally omitted.
