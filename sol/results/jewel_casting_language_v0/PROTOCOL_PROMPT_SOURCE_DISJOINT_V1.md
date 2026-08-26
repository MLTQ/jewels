# Prompt-to-Jewel source-disjoint binding protocol v1

## Question

Can a prompt-conditioned speaker emit native Jewel programs that retain prompt identity on a
held-out video when the same text prompt—but not the same pixels, latent, field, centroids, or
tokens—appears once in the training corpus?

This is the conventional in-distribution text-to-video feasibility question. It is distinct from
the stronger unseen style/action-combination scaling gate in
`PROTOCOL_PROMPT_DATA_SCALING_V1.md`. This protocol is frozen before fitting the three matched
training videos.

## Source ownership

Validation remains the three `train_00` videos and their three independent fitter replicas:

- `anime__05_ballet_train_00_seed60500`
- `cartoon__01_dogpark_train_00_seed61100`
- `render3d__04_workshop_train_00_seed62400`

The only matched-prompt additions are different LTX-generated source videos:

- `anime__05_ballet_train_01_seed60501`
- `cartoon__01_dogpark_train_01_seed61101`
- `render3d__04_workshop_train_01_seed62401`

No validation video or fitted validation field may enter training. The source IDs, seeds, paths,
and pixels differ; only their style/action text labels match.

## Frozen representation and speaker

- Keep the Gate 0f bundle-1 K=1024 codebook frozen.
- Emit one continuous centroid plus covariance, surface, and gradient token IDs for each Jewel.
- Train the existing factorized neural speaker with separate frozen BGE style/action embeddings,
  continuous centroid intensity NCE, and the existing token MLP.
- Use the exact Gate 1b optimizer, batch, generation, early-stopping, shuffled/null controls, and
  declared seed. The only change is the training field set: the complete 57-field grid plus the
  three source-disjoint matched-prompt videos (60 training videos total).

## Pass gate

Use the already registered Gate 1b conditions without relaxation:

1. Correct prompt beats both shuffled and prompt-blind null controls in centroid-density NCE.
2. Correct prompt beats both controls for every token role and by at least 2% macro NLL.
3. Independently sampled correct-prompt programs beat both controls by at least 0.02 target-field
   histogram cosine.
4. Three-way free-running prompt retrieval is at least 2/3.
5. Centers are not grid locked, renders are finite, and the inference audit contains only frozen
   style text, action text, and declared randomness.

Voxel PSNR remains diagnostic only. A pass establishes source-disjoint prompt binding, not novel
prompt composition, visual quality parity, or production readiness.

## Protocol-integrity result

**Invalid for the declared matched-prompt question.** After the run, direct manifest inspection
showed that the three `train_01` sources share only the semantic class, not the exact action text:

- anime ballet: `a dancer leaping across a stage` rather than the held-out pirouette sentence;
- cartoon dog park: `two dogs chasing each other in a park` rather than the held-out retriever
  sentence;
- 3D workshop: `a machinist turning metal on a lathe` rather than the held-out welder sentence.

The resulting report must therefore be labeled a **semantic-neighbor control**, not a pass or fail
of this protocol. Its 2/3 free-running retrieval is evidence of text-semantic signal, but it cannot
answer whether repeated exact-prompt sources defeat the null prior. A replacement protocol must
generate genuinely independent source videos from the exact held-out prompts and freeze their
source seeds before fitting.
