# Cell-local prefix continuation

## Outcome

Replacing the first model's single global prefix vector with aligned `16×16×8` context tokens
materially improves the learned continuation while keeping the model at 0.80M parameters. The
model still predicts only future jewel births; 18.6k–20.0k persistent jewels per view are copied by
stable ID with no learned update.

The matched 5,000-step CPU run took 490.7 seconds. Its correct-prefix predictions recover 99.95% of
the target birth count, with 0.331 mean absolute count error per cell and 0.0 carried-state error.

## Final control gate

The saved checkpoint was reevaluated at 16 identical sampled points per future frame. Each shuffled
prefix is disjoint from its evaluated target stride.

| Metric | Correct prefix | Shuffled prefix | Null prefix |
|---|---:|---:|---:|
| Standardized birth-feature MSE | **0.4475** | 1.1988 | 2.1695 |
| Birth count MAE per cell | **0.3310** | 3.7167 | 6.4994 |
| Sampled future-field PSNR | **19.870 dB** | 7.302 dB | -29.735 dB |

Correct context reduces mark error by 62.7% versus shuffled and 79.4% versus null. Relative to the
matched global-context model, correct mark MSE falls 35.6%, count MAE falls 72.7%, and rendered-field
PSNR improves by 3.41 dB.

The extreme null score is not a meaningful video baseline: this overfit never trained on dropped
context, so zero context emits physically unstable marks. A prompt-conditioned successor must use
explicit condition dropout so its unconditional branch is trained rather than assumed.

## Interpretation and limits

The matched comparison establishes that spatially aligned prefix information is essential. A
global vector can identify a window, but it cannot tell each future cell what local structure and
appearance should persist into it.

This remains an oracle-rank overfit to four views from one fitted clip. The model has not yet shown
held-out continuation, autonomous multi-stride rollout, or text control. Its visuals are
low-resolution field diagnostics rather than source-pixel reconstructions. The next gate is a small
multi-clip streaming corpus with explicit null/text conditioning and held-out source groups.

## Artifacts

- `continuation_controls.gif`: fitted future, correct prefix, disjoint shuffled prefix, null prefix
- `continuation_controls_contact.png`: start/middle/end frames from the same comparison
- `visual_report.json`: full low-resolution grid PSNR for the rendered view
- `summary.json`: authoritative denser four-view control metrics
- `train_log.jsonl`: training trace
- `continuation.pt`: local resumable model/optimizer checkpoint (ignored by Git; regenerate with the
  documented trainer when moving machines)
