# Four-class shared tokenizer: 3,000-step gate

## Protocol

This run repeats the leakage-safe four-class tokenizer experiment from the 500-step smoke with a
fresh learning-rate schedule: 12 source-group 1--3 windows train one shared `32^3` grid, 24-D
rank-conditioned tokenizer, while all four group-4 windows remain held out. Each fitted target has
120,000 jewels. Seventy-five percent of sampled render supervision comes from deterministic
training-source saliency pools split evenly between motion and saturated chroma.

The run completed 3,000 updates in 4,721.8 seconds (78.7 minutes) on the RTX 2070 SUPER. With batch
size one and uniform window sampling, each training window received approximately 250 direct
updates.

## Held-out result

The fixed 4,096-point audit reaches **16.541 dB mean/macro PSNR** and **96.19% count recovery**.

| Held-out source | PSNR | Decoded / target jewels |
|---|---:|---:|
| ApplyEyeMakeup group 4 | 14.853 | 113,837 / 120,000 |
| Basketball group 4 | 16.099 | 116,583 / 120,000 |
| HorseRiding group 4 | 15.445 | 116,439 / 120,000 |
| PlayingGuitar group 4 | 19.765 | 114,856 / 120,000 |

Validation peaked at 16.989 dB in the built-in 512-point audit at step 1,500 and ended at 16.506
dB. The higher-sample final score improves only 0.651 dB over the 500-step smoke and remains below
the 17.786 dB frozen Avenue-to-UCF transfer reference.

The visual gate **fails**. Background layout and broad palette improve, but the action-defining
subjects do not reliably survive: horse/rider and guitarist geometry collapse into texture, and
the basketball player is mostly lost. Prompt training is therefore paused because a prior cannot
restore semantic information removed by its tokenizer.

## Training-source diagnostic

Four source-group-1 windows seen during training reach **19.552 dB** and **98.63% count recovery**.
Their scenes are more recognizable than the held-out set but remain conspicuously noisy. This
3.012 dB train/validation gap shows a generalization problem, while the incomplete training
round-trips also implicate limited per-window exposure or the local set bottleneck itself.

A matched 1,000-update single-window overfit is the next discriminator. If it becomes clean, the
shared run was underexposed and should be trained with an exposure-matched schedule. If it remains
noisy, the rank/moment cell codec must be replaced before further corpus or prompt-prior scaling.

## Artifacts

- `heldout_eval_4096.json`: decisive held-out numerical audit
- `summary.json` and `train_log.jsonl`: complete optimization trajectory
- `manifest.json` and root `*_dense_roundtrip.gif`: four held-out visual comparisons
- `train_shard0/train_shard0_eval_4096.json`: seen-source numerical diagnostic
- `train_shard0/*_dense_roundtrip.gif`: four seen-source comparisons

The archived GIFs were generated immediately before the renderer provenance fix and their target
panels retain the legacy text `45k fitted target`; the manifests and checkpoints record the actual
120,000-jewel targets. `render_dense_tokenizer.py` now derives this label from each example. The
animations will be regenerated with corrected labels after the active overfit control releases the
2070S.

The 258 MB checkpoint remains on the compute host and is intentionally not tracked.
