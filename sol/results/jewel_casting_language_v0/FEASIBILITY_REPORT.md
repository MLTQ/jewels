# Native Jewel text-to-video feasibility report

> **Trajectory-language update (2026-08-26):** This report predates Gates 2a7–2b1. The flat and
> independent block failures below remain valid, but the next hierarchy has now produced a passing
> bounded prompt-only result: exact prompt text plus seed emits a two-token persistent trajectory
> program with 8/9 OpenCLIP retrieval, 9/9 correct-over-shuffled/null wins, recognizable
> ballerina/dog/welder output, exact 72k count, and zero grid locking. A learned 541k-parameter
> autoregressive speaker predicts 3/3 unseen-paraphrase scene tokens and 9/9 scene-consistent donor
> programs, although its raw rendered top-1 gate remains 4/9. See
> [`TRAJECTORY_SPEAKER_REPORT.md`](TRAJECTORY_SPEAKER_REPORT.md) for the current verdict and evidence.

## Executive verdict

The project has now proved that a useful irregular Jewel output language exists, but it has **not
yet proved text-to-video generation**.

The strongest representation result is a source-disjoint pass: exact continuous centroids plus
three K=1,024 active Jewel tokens (covariance, surface, gradient) reconstruct fresh fitted fields at
22.87 dB, preserve spacetime tilt at 1.042×, retain a 0.224 same-source language margin, and have
zero grid-center locking. That establishes that a model can speak a finite native Jewel vocabulary
without inheriting the earlier Cartesian output quantization.

Prompt experiments establish a narrower, still useful result. Correct text consistently beats
shuffled text, and adding one shared scene state plus increasing independent exact-prompt videos
from two to six moves the prompt-blind token margin from −0.152 to +0.034 and the free-generation
histogram margin from +0.0066 to +0.0160. The latter approaches, but does not cross, the frozen
+0.020 gate. Three-way retrieval is nonmonotonic (2/3, 2/3, 1/3), the final absolute gate fails,
and qualitative generations remain class-colored textures rather than recognizable subjects.

Therefore the honest pitch status is: **representation feasibility proved; bounded prompt signal
and favorable data scaling observed; generative scene structure not proved.** More data for the
current decoder alone is not the right next spend.

## What is proved

| Claim | Evidence | Verdict |
|---|---:|---|
| Fresh fields can be expressed by a finite native Jewel language | 22.8657 dB token-only reconstruction | pass |
| Time-distorted structure survives tokenization | mixed-tilt retention 1.0415 | pass |
| Output centroids need not sit on a Cartesian lattice | cell-center lock fraction 0.0000 | pass |
| The language is more canonical within a source than across sources | same-minus-different margin 0.2243 | pass |
| Sequence length is finite for a short window | about 35,265 active decisions per eight frames | pass, but expensive |
| Text changes source-disjoint generated Jewel distributions | correct repeatedly beats cyclic-shuffled controls | supported |
| Shared sample state helps free generation | correct-minus-null histogram margin −0.0097 → +0.0066 | causal improvement |
| More exact-prompt sources improve two continuous prompt metrics | token margin −0.152 → −0.035 → +0.034; histogram +0.0066 → +0.0126 → +0.0160 | supported |

## What is disproved for the current build

### Independent Jewel emission is not a scene generator

The first prompt speakers sample each centroid and mark independently conditional on text and
position. On both an unbalanced 63-field corpus and a balanced six-field exact-prompt corpus they
learn prompt-specific texture statistics, but the prompt-blind arm remains stronger and all free
renders collapse toward sparkling conditional averages. Longer training makes held-out correct and
shuffled loss worse after the 500-step checkpoint while null remains comparatively stable, so the
failure is not undertraining.

### One global scene vector is necessary but insufficient

A 32-dimensional stochastic scene state shared by all 72,000 emitted Jewels improves correct versus
shuffled likelihood and reverses correct versus null free-generation margin. With 18 exact-prompt
training videos it nearly reaches the histogram gate. It still cannot coordinate recognizable
objects or motion because every local Jewel remains independent after conditioning on that single
vector.

### The Gaussian text prior is not the principal remaining bottleneck

The preregistered posterior-oracle diagnostic gives the decoder a leaked training source posterior.
Against the correct text prior, it improves token NLL only 0.17% and histogram cosine only 0.0052,
failing the frozen 2% and 0.020 causal thresholds. Its qualitative output is still texture. This
localizes the next failure to the local decoder, not merely the text-to-scene prior.

## Why the original LLM-emits-Jewels idea can fail

1. **Permutation ambiguity.** A Jewel field is a set. A causal language model needs a stable order,
   but arbitrary sorting makes equivalent scenes different token sequences. Time gives a natural
   outer order; within a time window, local block order should own serialization rather than raw
   individual-Jewel order.
2. **Sequence length.** The proven language needs about 35k active token decisions for only eight
   frames before centroid precision is counted. Direct autoregression at that level wastes context
   and makes long continuation brittle.
3. **Conditional entropy.** A text prompt does not specify camera, identity, pose, background, and
   every motion detail. Independent maximum-likelihood emission averages those incompatible modes.
   Declared global randomness must choose one scene, then local state must keep every region
   consistent with it.
4. **Global-to-local bandwidth.** One 32-D vector cannot deliver all object boundaries, poses, and
   motion trajectories to tens of thousands of conditionally independent outputs. The posterior
   oracle directly demonstrates this bottleneck.
5. **Error accumulation.** If raw Jewels are autoregressive, one local mistake changes the context
   for thousands of later decisions. A hierarchy can resample or correct a local block without
   corrupting the whole window.
6. **Position precision.** Treating centroid bins as final coordinates recreates the grid artifact.
   Coarse cells may route attention, but emitted centroids must retain continuous offsets and the
   grid-lock audit must remain below 1%.
7. **Window seams.** A model that simply moves to the next volume can change subject, palette, or
   velocity at the boundary. Continuation must carry explicit overlapping block/trajectory state,
   not only the previous text prompt.

None of these objections rules out a Jewel-native generator. They rule out the flat utterance used
in the current prompt experiments.

## The next architecture: hierarchical Jewel utterance

The next speaker should emit three levels:

1. **Scene/window tokens** choose global mode: subject identity, style, camera, and broad motion.
2. **Local spacetime block tokens** describe a small irregular neighborhood and communicate with
   adjacent blocks. A 8×8×4 routing scaffold gives 256 block positions; it is an internal attention
   topology, not an output grid.
3. **Individual Jewels** emit continuous centroid offsets plus the already proven
   covariance/surface/gradient tokens, conditioned on the scene token, their block token, and
   neighboring block context.

The block sequence can be time-major and Morton-ordered within each time slab. This supplies a
stable causal order while keeping individual output centroids irregular. At continuation, the next
window receives the final overlapping time slab's block states and any Jewel trajectories crossing
the boundary.

## Gate 2: the next conclusive experiment

Use the existing 18-video exact-prompt corpus before generating more data.

1. Learn local block tokens only from training fields. Freeze their codebook and normalization.
2. Train a block-conditioned Jewel decoder. With oracle target block tokens, require recognizable
   structure and at least +0.02 histogram improvement over the scene-only decoder. If this fails,
   the block representation is insufficient and no prompt model should be trained.
3. Train a small text-conditioned block-token prior with correct, cyclic-shuffled, and null arms.
   The final program must contain no video-, source-, field-, or class-ID input.
4. Require the existing irregularity and finite-render checks, at least 2/3 prompt retrieval across
   multiple registered generation seeds, and a blinded recognizable-subject panel for at least two
   of the three prompts.
5. Only after the oracle block decoder passes should exact-prompt source count be expanded. That is
   the point where additional data and compute have a demonstrated mechanism to improve.

## Pitch threshold

The project becomes pitchable as “given more data and compute, this can become text-to-video” when
all of the following are simultaneously true:

- the Gate 0f native Jewel language remains passed;
- prompt-only hierarchical generations beat shuffled and null controls on held-out sources;
- at least two of three exact prompts produce recognizable subject/action structure rather than
  palette-only texture;
- the result improves on a preregistered data curve without changing thresholds;
- a second window continues identity and motion using only generated carry state.

The current branch is one architecture gate short of that milestone. Its evidence is strong enough
to justify building the hierarchical block-token speaker, but not yet strong enough to make the
full compute-scaling pitch.

## Evidence artifacts

- `hierarchical_v1/gate0f_individual/individual_language_evidence.png`
- `hierarchical_v1/gate0f_individual/qualitative.png`
- `prompt_training_curves.png`
- `prompt_exact_repetition_v1/aggregate/exact_prompt_repetition.png`
- `exact_prompt_sources_v2/source_provenance.png`
- `prompt_shared_scene_v1/ablation/shared_scene_ablation.png`
- `prompt_shared_scene_scaling_v1/aggregate/shared_scene_data_scaling.png`
- `prompt_shared_scene_scaling_v1/r6/qualitative.png`
- `prompt_shared_scene_scaling_v1/posterior_oracle/qualitative.png`
