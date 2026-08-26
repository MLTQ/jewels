# Gate 2b1 protocol: learned text-to-trajectory-token speaker

Frozen after Gate 2b0 passed prompt-only semantic and safety gates with an exact prompt compiler.

## Decision question

Once the native grammar owns window coherence, can a small learned autoregressive speaker replace
the lookup and emit valid `scene -> foreground trajectory -> background` programs from unseen text
paraphrases?

## Vocabulary and text split

- Three semantic scene tokens and the same 18 source-backed trajectory tokens, six per scene.
- Each scene has its exact Gate 2b0 prompt plus two authored training paraphrases and one disjoint
  evaluation paraphrase. Prompt lists are checkpointed verbatim.
- Training pairs include every ordered pair of distinct same-scene donor tokens except the six
  cyclic `(i, i+1 mod 6)` pairs. Those 18 pairs are held out and evaluated only with the unseen
  paraphrase.
- Text embeddings are frozen normalized OpenCLIP ViT-B/32 `laion2b_s34b_b79k`. Empty-text dropout
  is 10% during training so the null branch is learned rather than invented at audit time.

## Model and optimization

- A 256-wide text MLP predicts the scene token. Foreground logits condition on text plus the scene
  embedding; background logits condition on text, scene, and the emitted foreground token.
- Source logits are unmasked by scene. Only the syntactically invalid repeated background token is
  masked while sampling. Thus scene-consistent donors remain a learned result.
- Generation uses scene argmax and categorical foreground/background sampling at temperature 0.8
  from the top six logits, with the registered seed.
- AdamW, learning rate 0.002, batch 64, seed 20260917, maximum 5,000 updates.
- Evaluate every 100 updates. After update 1,000, stop after ten consecutive evaluations without a
  correct held-out NLL improvement of at least 1e-4. Save every evaluation and retain the best
  correct-NLL checkpoint.

## Token gate

1. Best held-out correct token NLL is at least 20% below cyclic-shuffled text and at least 10% below
   learned null text.
2. Scene top-1 is 3/3 on unseen paraphrases.
3. Across generation seeds 20260918, 20260919, and 20260920 per paraphrase, at least 8/9 sampled
   programs emit both donors from the predicted scene; all foreground/background tokens are
   distinct.

## Rendered gate

Render learned correct, cyclic-shuffled, and null programs with the exact Gate 2b0 realizer and
OpenCLIP battery.

1. Correct programs retrieve the intended prompt at least 6/9, with majority retrieval in at least
   two classes.
2. Mean intended-prompt cosine beats shuffled generation by at least 0.02 and null by at least 0.01;
   correct wins against shuffled at least 7/9.
3. Every field has exact 50/50 distinct donor ownership, 72,000 Jewels, finite output, and below 1%
   grid locking.
4. At least two prompt classes are recognizable in at least two seeds and visibly change under
   cyclic-shuffled text.

## Claim boundary

A pass shows that learned text-to-token emission is not the current bottleneck and supports the
“more vocabulary/data/compute” pitch. The model remains a small three-prompt proof with
source-backed macro tokens, not an open-vocabulary foundation model.
