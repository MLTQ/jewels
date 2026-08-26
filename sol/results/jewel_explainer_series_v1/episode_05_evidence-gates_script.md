# Episode 5: What the experiments actually prove

Exact gate, learned speaker, plateau, and honest failure scope

## Claim sources

- `sol/audit_prompt_trajectory_speaker.py`
- `sol/train_learned_trajectory_speaker.py`
- `sol/audit_learned_trajectory_speaker.py`
- `sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md`

## 1. Freeze the gate before looking

The exact audit freezes prompts, seeds, field size, renderer, frame indices, semantic evaluator, and pass thresholds before execution. It renders frames zero, twenty-four, and forty-eight at one-forty-four by two-sixteen. OpenCLIP ViT-B thirty-two embeds each frame, normalized frame embeddings are mean-pooled, and the resulting video vector is compared against the three exact prompts. The preregistration prevents a visually appealing seed from redefining success.

**On screen:** 3 prompts × 3 seeds × 3 causal conditions; frozen before execution.

## 2. Retrieval and generation margins test different things

Top-one retrieval asks whether a correct generated field is closest to its intended prompt among three choices. The shuffled generation margin instead holds the intended text fixed and compares a correctly conditioned field against a field actually generated from wrong text. Null margin does the same with no semantic prompt. Pairwise win counts retain seed-level information that a mean could hide. All are needed because any one semantic score can be fooled by class imbalance or evaluator bias.

**On screen:** retrieval: rank text for one field; margin: change the generated field itself.

## 3. Exact Gate two-b-zero passes

Across nine exact prompt programs, intended retrieval succeeds eight times. Every prompt class has majority retrieval. Mean correct-minus-shuffled generation similarity is plus zero-point-zero-five-six-four-eight, above the frozen plus zero-point-zero-two threshold. Correct-minus-null is plus zero-point-zero-three-three-four-eight, above plus zero-point-zero-one. Correct wins all nine shuffled and all nine null pairs. Counts, donor distinction, finite rendering, and continuous-center checks also pass.

**On screen:** 8/9 retrieval · +0.05648 shuffled · +0.03348 null · 9/9 wins

## 4. The learned speaker is deliberately small

Gate two-b-one replaces exact lookup with a five-hundred-forty-one-thousand-two-hundred-twenty-three-parameter autoregressive network. Frozen text embeddings enter a two-layer projection. A scene head predicts one of three semantic tokens. Scene embedding conditions the foreground head. Text, scene, and sampled foreground condition the background head. Source logits are not masked by scene; only exact foreground-background repetition is forbidden. Scene-consistent donors must therefore be learned rather than imposed.

**On screen:** text → scene → foreground → background; only donor repetition is masked.

## 5. Held-out wording and held-out donor pairs

Training uses two authored paraphrases plus the exact prompt for each class and enumerates ordered donor pairs, while reserving cyclic pairs for evaluation. The held-out set changes both wording and foreground-background combination. Ten percent empty-text dropout teaches a null condition. Correct, cyclic-shuffled, and empty embeddings score the identical target programs, which makes token negative log likelihood a clean conditional test before any renderer or CLIP model enters the loop.

**On screen:** Evaluation changes both paraphrase and donor combination.

## 6. Longer training did not solve the learned gate

The best correct-condition negative log likelihood occurs at step one hundred. Evaluation continues every one hundred updates. Ten consecutive evaluations through step eleven hundred fail to improve it, satisfying the frozen plateau rule. Correct held-out NLL is two-point-two-zero-zero-five, compared with four-point-three-eight-seven under cyclic-shuffled wording and two-point-five-three-nine-one for null. Scene accuracy is one hundred percent for correct unseen paraphrases. This is evidence of early overfit, not evidence that the run was simply too short.

**On screen:** best step 100; ten stale evaluations; stopped at 1,100 by the frozen rule.

## 7. The learned result is a scoped near-pass

All nine correct-text samples predict the right scene and emit both unmasked donors from that scene. Rendered correct-minus-shuffled and correct-minus-null margins pass, and correct beats shuffled in seven of nine. But raw three-way OpenCLIP top-one is only four of nine, below the required six, with majority retrieval in only one class. We retain that failure. The learned network understands the tiny program syntax and causal scene conditioning; it does not yet establish robust open-vocabulary semantic generation.

**On screen:** Token syntax and causal margins pass; strict learned retrieval remains 4/9 and fails.
