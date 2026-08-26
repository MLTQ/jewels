# Episode 5: How we decide whether it really works

Tests designed to resist wishful thinking

## Claim sources

- `sol/audit_prompt_trajectory_speaker.py`
- `sol/train_learned_trajectory_speaker.py`
- `sol/audit_learned_trajectory_speaker.py`
- `sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md`

## 1. Write the rules before the run

Before generating results, we freeze the prompts, seeds, renderer, sampled frames, scoring method, and pass thresholds. This is called preregistration: deciding what counts as success before seeing which samples look good. We test three prompts, three seeds, and three prompt conditions. An automatic text-image scorer compares three frames from each video with the intended text.

**On screen:** Freeze the test and pass line before looking at the results.

## 2. Ask two different questions

The first test asks whether a generated video matches its intended prompt better than the other two prompt labels. The second asks whether correct text produces a better video than wrong or empty text, while keeping the intended meaning fixed for scoring. The first checks recognition. The second checks causal control. We also count individual wins so a good average cannot hide repeated failures.

**On screen:** Recognition ranks labels; causal testing compares separately generated videos.

## 3. The exact recipe passes

Across nine videos made by the exact recipe writer, eight were matched to the intended prompt. Correctly prompted videos beat wrong-prompt videos in all nine comparisons and beat empty-prompt videos in all nine. The average margins cleared the thresholds we set in advance. The field-size, rendering, and irregular-position checks also passed.

**On screen:** 8 of 9 recognized · 9 of 9 beat wrong prompts · 9 of 9 beat empty prompts

## 4. Then replace rules with learning

The exact writer uses hand-built prompt lookup. Our next test replaced it with a small learned network containing about five hundred forty-one thousand adjustable numbers. The network reads a text embedding, which is a numerical summary of the sentence. It predicts the scene first, then the subject source, then the setting source. Only choosing the exact same source twice is forbidden; scene-consistent choices must be learned.

**On screen:** text summary → scene choice → subject choice → setting choice

## 5. Test unfamiliar wording and pairings

We hold back both a new paraphrase and specific subject-setting combinations. Held out means the learner does not see those examples during training. At evaluation, it must understand different wording and choose a source pair it did not memorize as a pair. Empty-text examples teach the network what no prompt looks like. This makes the test stricter than repeating the training phrases.

**On screen:** Evaluation changes both the wording and the source pairing.

## 6. More training reached a plateau

We continued training and measured the held-out error every one hundred steps. The best result appeared at step one hundred. Ten later measurements, through step eleven hundred, did not improve it. That flat stretch is a plateau: more of the same training was no longer helping. The network separated correct wording from wrong wording, but it also began fitting the tiny training set too closely.

**On screen:** Best at step 100; no improvement through step 1,100.

## 7. Keep the near-pass honest

The learned writer chose the correct scene for all nine held-out prompts and usually beat the causal controls. But only four of nine raw videos were matched to the right prompt, below our required six. So the learned gate fails. The useful conclusion is narrower: the network learned the tiny recipe syntax and responded to text, but the data and vocabulary were too small for robust prompt-to-video meaning.

**On screen:** The learned writer responds to text, but strict recognition is only 4 of 9.
