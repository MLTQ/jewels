# Gate 2b0 protocol: prompt-only trajectory-token speaker

Frozen after the coherence ceiling restored all three subjects and both two-donor trajectory
oracles preserved recognizable prompt-specific compositions, while their low-level histogram/count
gates rejected the hand-designed masks.

## Decision question

Can text and declared randomness alone emit a finite native Jewel program that produces a
recognizable, prompt-controlled, non-grid-locked video without an input video, target field, target
block program, or held-out latent?

## Frozen speaker

- Vocabulary: three semantic scene tokens; six foreground trajectory tokens per scene; six
  background tokens per scene; the frozen three K=1,024 active Jewel vocabularies.
- Exact registered prompts are the three scene keys already used by the 18-source corpus:
  ballerina pirouette, golden retriever catching a ball, and welder joining steel.
- The prompt lookup emits the semantic scene token. For seeds 20260914, 20260915, and 20260916, a
  deterministic permutation of the six eligible training programs emits distinct foreground and
  background tokens. No target program participates.
- The training-only semantic path from Gate 2a10 ranks each donor's Jewels by squared distance from
  the moving tube. Cast the closest 36,000 foreground Jewels and the farthest 36,000 background
  Jewels. This rank-balanced rule is frozen before execution because Gate 2a10 showed that one
  shared geometric radius can miss the total count when two valid decompositions have different
  local densities. It uses neither target rows nor render metrics.
- This is a template-backed generative vocabulary, not retrieval of a complete video: the two
  source tokens each own exactly half the emitted field.
- Cyclic-shuffled control compiles the next prompt under the matched seed. Null control chooses one
  of the three scene tokens from seed alone and then emits its donor pair.

## Frozen semantic evaluation

- Render frames 0, 24, and 48 at 144x216 with the support-correct Jewel renderer.
- Mean-pool normalized OpenCLIP ViT-B/32 `laion2b_s34b_b79k` image embeddings over the three frames.
- Compare with normalized embeddings of the three exact prompts and the empty string.
- Report correct-prompt top-1 retrieval, correct-field similarity to intended text, similarity of
  the cyclic-shuffled/null fields to that same intended text, and all per-seed margins.

## Gate

1. Correct fields retrieve their intended prompt for at least 6/9 programs, with majority retrieval
   in at least two of three prompt classes.
2. Mean intended-prompt cosine beats cyclic-shuffled generation by at least 0.02 and null generation
   by at least 0.01; correct wins pairwise in at least 7/9 cases.
3. Every program has distinct donors, exactly 50% contribution from each, zero count adjustment,
   exactly 72,000 output Jewels, finite renders, and less than 1% grid locking.
4. At least two prompt classes are recognizable in at least two of three seeds. Cyclic-shuffled
   rows must visibly change the subject class.

## Claim boundary

A pass is conclusive proof of a **promptable native Jewel action language** on three registered
prompts and a mechanism-backed reason that more vocabulary/data/compute can extend it. It is not an
open-vocabulary learned LLM, and source-level tokens remain backed by training constellations. The
next gate replaces those tokens with learned object/trajectory prototypes and prompt-to-token
prediction.
