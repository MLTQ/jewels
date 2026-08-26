# Shared-scene exact-prompt data scaling Gate 1h

## Decision question

Does the shared-scene native Jewel speaker cross the frozen prompt-binding gate when exact-prompt
source diversity increases from two to four to six independent videos per prompt?

This protocol is frozen after Gate 1g's two-source result failed the absolute gate but improved
correct free-run histogram match over both shuffled and prompt-blind controls.

## Frozen source expansion

Generate four additional 49-frame, 768x512 LTX-2.3 videos for each validation prompt with the same
style suffix, FP8-cast quantization, CPU offload, stable-shot contract, and one sample per invocation.
Only source seeds change:

| Style | Exact action text | New seeds |
|---|---|---|
| anime | `a ballerina spinning a pirouette in a studio` | 71002, 71003, 71004, 71005 |
| cartoon | `a golden retriever catching a ball on grass` | 72002, 72003, 72004, 72005 |
| render3d | `a welder joining steel with bright sparks` | 73002, 73003, 73004, 73005 |

Fit every completed source once with the unchanged support-correct 3,000-step, 72,000-Jewel
teacher contract. No original held-out validation source may enter training.

## Nested training points

Keep the Gate 1g model, 32-dimensional shared scene, KL weight 0.05, prior-use probability 0.25,
frozen BGE/codebook, optimization, prompt dropout, checkpoint selection, generation settings, and
controls unchanged.

1. **r2:** the already completed two-video-per-prompt Gate 1g result.
2. **r4:** add seed suffixes 02 and 03 for each prompt (12 training fields total).
3. **r6:** add suffixes 04 and 05 as well (18 training fields total).

## Frozen decision

The Gate 1g absolute checks remain unchanged at r6. Call data scaling positive only if the
correct-versus-null token-NLL margin, correct-versus-null free-run histogram margin, and retrieval
accuracy are all nondecreasing from r2 to r4 to r6. Report every point even if an earlier point
fails. A pass proves bounded-vocabulary source-disjoint prompt generation with shared global state;
it does not prove arbitrary prompts, high fidelity, or persistent multi-window continuation.
