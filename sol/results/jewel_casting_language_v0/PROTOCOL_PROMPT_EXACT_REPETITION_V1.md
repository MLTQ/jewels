# Exact-prompt source-repetition Gate 1e protocol

## Decision question

Does the frozen native Jewel speaker defeat its prompt-blind prior when each validation prompt has
one and then two genuinely independent training videos generated from the exact same text?

This replaces the invalid matched-prompt assumption recorded in
`PROTOCOL_PROMPT_SOURCE_DISJOINT_V1.md`. It is frozen before any of the six new source videos are
generated.

## Frozen source generation

Generate two new 49-frame, 768x512 LTX-2.3 videos for each exact held-out prompt. Keep the original
style suffix, FP8-cast quantization, CPU offload, stable-camera/continuous-shot contract, and one
sample per invocation. Only the declared source seed changes:

| Style | Exact action text | New seeds |
|---|---|---|
| anime | `a ballerina spinning a pirouette in a studio` | 71000, 71001 |
| cartoon | `a golden retriever catching a ball on grass` | 72000, 72001 |
| render3d | `a welder joining steel with bright sparks` | 73000, 73001 |

Fit every completed video independently with the existing support-correct 3,000-step, 72,000-Jewel
contract. Seed-suffix `00` fields form repetition point one; suffix `01` fields are added for point
two.

## Frozen validation ownership

Validation remains the original three `train_00` source videos, each with fitter seeds 0, 1, and 2:

- `anime__05_ballet_train_00_seed60500`
- `cartoon__01_dogpark_train_00_seed61100`
- `render3d__04_workshop_train_00_seed62400`

None of those pixels, fields, centroids, tokens, source IDs, or generation seeds may enter the
speaker. The new videos share text only.

## Nested speaker points

Keep the Gate 1b bundle-1 K=1,024 codebook, factorized neural speaker, frozen BGE style/action
embeddings, optimizer, prompt dropout, early stopping, sampling budgets, free-generation seed, and
correct/cyclic-shuffled/null controls unchanged.

1. **r1:** the 57-video compositional base plus one exact-prompt source per validation prompt (60
   training videos).
2. **r2:** the same base plus both exact-prompt sources per validation prompt (63 training videos).

## Frozen Gate 1e at r2

All Gate 1b checks remain required:

1. Correct density NCE beats shuffled and prompt-blind null.
2. Correct NLL beats both controls for covariance, surface, and gradient; correct macro token NLL
   improves at least 2% over the better control.
3. Correct free-running histogram cosine beats both controls by at least 0.02.
4. Correct-prompt three-way retrieval is at least 2/3.
5. Centers are not grid locked, renders are finite, and inference receives only style text, action
   text, and declared randomness.

Report r1 even if it fails. Call repetition scaling positive only when correct-versus-null density
and token margins, correct free-run histogram match, and retrieval are all nondecreasing from r1
to r2. A pass proves source-disjoint in-distribution prompt binding, not novel prompt composition or
production visual quality.
