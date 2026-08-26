# Prompt-to-Jewel data-scaling protocol v1

## Question

Does prompt binding improve predictably when the frozen individual-Jewel language is trained on
broader source coverage, or is the weak correct-versus-shuffled signal observed at 33 training
videos merely noise?

This protocol is frozen before fitting the final 24 videos or evaluating the larger point.

## Frozen representation and validation set

- Keep the Gate 0f bundle-1 K=1024 codebook frozen.
- Emit one exact continuous centroid and three discrete marks per Jewel: covariance, surface, and
  gradient. There is no continuous appearance residual.
- Hold out all three independent fitter replicas of these source videos from every prompt model:
  - `anime__05_ballet_train_00_seed60500`
  - `cartoon__01_dogpark_train_00_seed61100`
  - `render3d__04_workshop_train_00_seed62400`
- The held-out style/action combinations must remain absent from the training fields. Their style
  and action factors may occur separately in other combinations.

## Nested data points

The same preregistered additive style/action caster is evaluated without parameter tuning at:

1. **9 sources:** the original balanced 12-source tranche minus the three held-out sources.
2. **33 sources:** the existing 36-source corpus minus the three held-out sources.
3. **57 sources:** the complete balanced 60-combination `train_00` grid minus the same three
   held-out combinations. The missing 24 fields use the existing support-correct 3,000-step,
   72,000-Jewel fitter contract.

Every point uses 8,192 Jewels per training source, 16,384 per validation fit, 72,000 generated
Jewels, 4,096 render probes, cell concentration 256, token concentration 64, and seed 20260902.

## Controls and metrics

For each data point report:

- teacher-forced cell NLL and covariance/surface/gradient token NLL for the correct prompt,
  cyclically shuffled prompt, and prompt-blind null prior;
- correct-prompt token macro improvement over the better control;
- free-running target-histogram cosine for correct, shuffled, and null prompts;
- three-way retrieval accuracy from independently sampled correct-prompt programs;
- exact frozen-text factor resolution, grid-lock fraction, finite renders, and an inference-input
  audit proving no target field, target centroid, target token, source pixel, or source identity is
  available at generation time.

Voxel PSNR is diagnostic only because the free sample is stochastic and set correspondence is not
identified.

## Frozen pass gate at 57 sources

The final point passes only if all of the following are true:

1. Correct-prompt cell NLL beats both shuffled and null controls.
2. Correct-prompt NLL beats both controls for all three token roles.
3. Correct token macro NLL improves by at least 2% over the better control.
4. Correct free-running histogram cosine beats both controls by at least 0.02.
5. Correct free-running retrieval is at least 2/3.
6. Frozen-text factor resolution is exact, center grid-lock is zero, renders are finite, and the
   inference audit is prompt-only.

## Scaling conclusion

Independently of the absolute gate, call the result **positive data scaling** only if both of these
families improve monotonically from 9 to 33 to 57 sources:

- correct-versus-shuffled cell and token-macro NLL margins;
- correct free-running target-histogram cosine and retrieval accuracy.

The null comparison must be shown at every point. A larger model is not introduced during this
curve, so improvement cannot be attributed to extra model capacity or longer optimization.
