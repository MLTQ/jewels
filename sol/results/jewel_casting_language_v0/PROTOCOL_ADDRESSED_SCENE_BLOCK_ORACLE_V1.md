# Gate 2a7 protocol: address-conditioned scene/block constellation oracle

Frozen after the fine-routing review and before implementation or execution.

## Corrected contract

Every previous block language serialized an explicit block address, and the neural expander used
that address. The empirical and medoid constellation realizers accidentally ignored it when
selecting a predefined constellation: a token learned in one region could be transplanted into any
other region. The resulting mosaics are therefore evidence about an address-free token lookup, not
the proposed `(scene, address, local token)` grammar.

Gate 2a7 tests the intended grammar directly.

## Frozen experiment

- Reuse the 16x16x8/K=1024 fine block language, its 2,048-token programs, the three semantic scene
  tokens plus one pooled null, the 18/9 split, and every Gate 2a6 evaluation setting.
- For each addressed program token, choose its complete medoid only among the six training block
  constellations with the **same address** and semantic scene token. The null scene chooses among
  all 18 same-address constellations.
- Choose the medoid by minimum squared distance between the frozen K=1024 prototype and the
  candidate block's normalized 77D descriptor.
- Pool the four nearest same-address candidates for role likelihood with smoothing 0.1; generation
  casts the nearest complete constellation with local jitter 0.005.
- Assemble all 2,048 addressed constellations and report the exact-72k count adjustment. Evaluate
  histograms on the original 8x8x4 comparison grid and grid locking on both routing/evaluation grids.

Controls independently shuffle the scene token, shuffle the block-token sequence while retaining
addresses, or replace both with the pooled-null scene and most frequent token. Randomness is matched.

## Gate

Use every unchanged Gate 2a6 numerical and safety threshold. Qualitatively, at least two of three
source-disjoint oracle rows must show a recognizable/localized prompt-consistent subject or scene
structure that is absent from shuffled-scene, shuffled-block, and null rows.

If this passes, the finite hierarchical Jewel grammar exists and Gate 2b may learn the scene token
and addressed block sequence from text. If it fails qualitatively, add explicit object/track tokens;
do not draw a broader conclusion from the address-free realizers or train them longer.
