# Promptable native Jewel trajectory language: proof report

## Verdict

The branch now contains a bounded proof that text and declared randomness can generate a
recognizable video by speaking a finite native Jewel program into an irregular spacetime volume.
It is not yet a general text-to-video model.

The passing Gate 2b0 system receives only one of three registered prompt strings and an integer
seed. It emits a semantic scene token, one persistent foreground-trajectory token, and one distinct
background token. Those tokens cast exactly 72,000 continuous-centroid Jewels using the frozen
three-way K=1,024 covariance/surface/gradient vocabulary. There is no input video, target field,
target block program, held-out latent, Cartesian output grid, or complete-video retrieval.

Across nine programs, intended-prompt OpenCLIP retrieval is 8/9; correct programs beat matched
cyclic-shuffled and null generations in 9/9 cases. Mean correct-minus-shuffled and
correct-minus-null margins are `+0.05648` and `+0.03348`, above the frozen `+0.02` and `+0.01`
gates. All three prompt classes have majority retrieval, every program combines two distinct
training-backed macro tokens at exact 50/50 ownership, every output contains exactly 72,000 Jewels,
and grid-center locking is zero. Visual review shows a ballerina, a dog, and a workshop/welder and
shows the subject changing under cyclic-shuffled text.

![Causal and quantitative evidence](trajectory_speaker_evidence_v1/trajectory_speaker_evidence.png)

![Matched correct-versus-shuffled middle frames](trajectory_speaker_evidence_v1/trajectory_speaker_proof_sheet.png)

## What changed the result

The physical Jewel vocabulary was not the limiting factor. The factorization was.

| Architecture | Key result | Recognizable classes / 3 | Decision |
|---|---:|---:|---|
| Independent global scene decoder | Posterior token NLL `6.1234`; texture | 0 | reject |
| Addressed 16x16x8 local phrases | Direct NLL `5.1680`, 15.60% better than global; texture | 0 | local statistics pass, generation fails |
| One coherent source program | Source-disjoint numeric controls pass; complete subject structure returns | 3 | coherence mechanism passes; retrieval-only |
| Two-donor trajectory tube | New foreground/background composition; wrong-object control swaps class | 3 | qualitative pass; frozen count/histogram gate fails |
| Exact prompt trajectory speaker | `8/9` retrieval; `9/9` shuffled/null wins | 3 | all Gate 2b0 checks pass |
| Learned unseen-paraphrase speaker | `3/3` scene prediction; `9/9` scene-consistent programs; causal render margins pass | 3 | one OpenCLIP top-1 gate fails (`4/9 < 6/9`) |

The decisive causal comparison is addressed local phrases versus one persistent source choice. Both
use the same continuous centroids and active K=1,024 physical tokens. Independent choices produce
texture; one cross-window owner restores the subject. A later two-donor experiment then retains the
subject while combining a foreground tube and background from distinct source programs, ruling out
complete-field retrieval as the only coherent mode.

## Learned speaker result

A 541,223-parameter autoregressive speaker replaces the exact lookup with
`text -> scene -> foreground token -> background token`. Source logits are not masked by scene; only
exact donor repetition is syntactically prohibited.

It trains on 216 programs using exact prompts plus two paraphrases per class and evaluates 18
unseen cyclic donor pairs under a fourth, unseen paraphrase. The best checkpoint occurs at step 100.
Ten subsequent evaluations through step 1,100 fail to improve held-out correct NLL, meeting the
frozen plateau rule and demonstrating overfit rather than undertraining.

| Held-out condition | Token NLL | Scene accuracy |
|---|---:|---:|
| correct unseen paraphrase | **2.2005** | **100%** |
| cyclic-shuffled paraphrase | 4.3870 | 0% |
| learned null | 2.5391 | 33.3% |

All nine learned correct-text samples emit both unmasked donors from the predicted scene. At render
level, correct-minus-shuffled is `+0.04782`, correct-minus-null is `+0.04117`, and correct beats
shuffled in the required `7/9`. The strict learned gate nevertheless remains failed because raw
three-way OpenCLIP top-1 retrieval under the authored evaluation paraphrases is `4/9`, with majority
retrieval in only one class. Most ballerina and dog renders are ranked as the broadly worded
workshop prompt despite visible class separation. This failure is retained; the stronger causal
margins and token consistency do not retroactively change the gate.

## What is proved

1. A finite active Jewel vocabulary can render fresh irregular fields without position
   quantization: Gate 0f remains `22.8657 dB`, zero grid locking, and 1.0415x tilt retention.
2. Persistent cross-block/temporal ownership is sufficient to restore recognizable subjects with
   that vocabulary.
3. Two source-backed macro tokens can be cast into one new field while retaining prompt-consistent
   structure, so the working mechanism is compositional rather than complete-video retrieval.
4. Text and declared randomness alone can choose and cast those programs with strong causal prompt
   controls.
5. A small learned autoregressive network can predict scene and valid source-level tokens from
   unseen paraphrases; learning the token syntax is not the main bottleneck.

## What is not proved

- The vocabulary has only three semantic prompts and 18 source-backed macro tokens.
- Foreground/background tokens still expand to constellations taken from training fields. They are
  predefined generative primitives, not learned reusable object/pose/velocity prototypes.
- The semantic trajectory is learned as one class-level path, not emitted freely as object motion.
- Local quality remains soft and sparkly; the result is recognizable proof, not promotional visual
  quality.
- No second generated window yet carries identity and velocity forward.
- Open-vocabulary composition, multiple objects, camera control, counting, and relationships remain
  untested.

## Feasibility interpretation

The original flat “LLM emits Jewels” proposal was under-specified. A language model should not emit
independent primitive Jewels or 2,048 independent block phrases. It should speak a typed,
hierarchical action language:

```text
prompt + seed
  -> scene/style/camera tokens
  -> persistent object and background trajectory tokens
  -> trajectory anchors and derivatives through the window
  -> addressed local constellation tokens
  -> continuous-centroid physical Jewel tokens
```

This is technically feasible because the branch now has a passing example of every boundary except
the replacement of source-backed macro tokens by reusable learned prototypes. The evidence also
identifies why more data and compute help: they buy a larger object/trajectory vocabulary, better
prototype learning, and a program prior with broader prompt coverage. They should not be spent on a
larger independent-Jewel decoder.

## Next conclusive experiment

Train 64–256 reusable foreground/background trajectory prototypes over at least 100 prompted fields
from 10–20 compositional prompts. Each prototype should own a connected spacetime tube but decode
through the existing continuous-centroid physical vocabulary. Train a small program transformer to
emit scene, object, path-derivative, and background tokens.

Freeze the next gate before corpus generation:

1. held-out prompts must use object/action combinations absent from training;
2. correct text must beat cyclic object/action swaps at rendered CLIP and blinded recognition;
3. no emitted macro token may reference a source ID or copy more than a registered fraction of one
   source field;
4. at least two independently sampled programs per prompt must be recognizable and structurally
   different;
5. a second window must continue a generated trajectory using only carried program/Jewel state.

That experiment converts the current bounded proof into the pitch claim: quality and prompt breadth
improve as reusable trajectory vocabulary, paired data, and program-model compute scale.

## Evidence inventory

- `prompt_trajectory_speaker_v1/report.json`: passing exact prompt-only numeric gate.
- `prompt_trajectory_speaker_v1/qualitative_seed*.png`: three exact-prompt causal seed sheets.
- `learned_trajectory_speaker_v1/report.json`: learned token training and plateau report.
- `learned_trajectory_speaker_v1/progress.json`: every 100-update condition curve.
- `learned_trajectory_speaker_v1/audit/report.json`: learned unseen-paraphrase rendered gate.
- `learned_trajectory_speaker_v1/audit/qualitative_seed*.png`: learned causal seed sheets.
- `trajectory_speaker_evidence_v1/trajectory_speaker_evidence.png`: consolidated graph.
- `trajectory_speaker_evidence_v1/trajectory_speaker_proof_sheet.png`: matched qualitative proof.
