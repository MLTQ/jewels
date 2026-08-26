# Episode 4: Why motion needs a persistent owner

The experiments that turned texture into subjects

## Claim sources

- `sol/block_token_jewel_speaker.py`
- `sol/coherent_source_realizer.py`
- `sol/semantic_trajectory_realizer.py`
- `sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md`

## 1. Good pieces can make a bad whole

Our first generator chose each local Jewel from the prompt and its nearby position. Individual patches often had plausible colors and textures, yet the whole video had no recognizable subject. It was the visual version of assembling a jigsaw from individually convincing pieces taken from different boxes. Local correctness does not automatically create one object that survives across time.

**On screen:** Plausible local pieces do not guarantee one coherent subject.

## 2. Larger patches still lacked identity

We next predicted small blocks of Jewels instead of single ones. The model became better at predicting which block belonged in each neighborhood, but the rendered result was still texture. That was useful evidence: the local statistics were improving, while identity was still missing. Something had to tell distant regions and later frames that they belonged to the same subject.

**On screen:** Better local prediction still did not create a persistent subject.

## 3. One owner brought subjects back

As a diagnostic test, we assigned one complete training-owned program to the entire spacetime window. The physical Jewel vocabulary did not change, but ballerina, dog, and welder subjects became recognizable again. This test was retrieval, not new generation. Its value was causal: it showed that persistent ownership across blocks and frames was the missing ingredient.

**On screen:** Same Jewels, one persistent owner: recognizable subjects return.

## 4. Mix two sources to test composition

A complete training program could merely replay one example. To test that, we built a new field from two different programs. One supplied Jewels near the moving subject tube; the other supplied the surroundings. When we deliberately swapped in a wrong subject source, the visible subject changed. The result was not a replay of either complete source, and the tube truly controlled subject identity.

**On screen:** A moving subject and a separate setting form a new field.

## 5. Find where the subject differs

The subject path comes from a saliency map. Saliency simply means a map of what stands out. For each time step, we compare the chosen scene with the other scenes and find the region where their learned descriptions differ most. A small smoothing pass prevents the path from jumping. This gives us a class-related trajectory without looking at the target test video.

**On screen:** A difference map finds the subject region, then smoothing connects the path.

## 6. Choose by rank, not radius

Different fields can pack Jewels at different densities. A fixed tube width might collect too many from one source and too few from another. Instead, we sort by distance from the path. We take the closest thirty-six thousand subject Jewels and the farthest thirty-six thousand setting Jewels. Ranking guarantees the requested count without duplicates or target-specific tuning.

**On screen:** Sort by tube distance to get an exact, balanced field.

## 7. Change the cause, not just the label

A pretty sample is not enough. For every prompt and seed, we generate three fields: one from the correct prompt, one from a deliberately wrong prompt, and one with no prompt meaning at all. These are causal controls: we change the input that should cause the result, then check whether the generated field changes in the intended direction. The video itself is regenerated for every condition.

**On screen:** Correct, wrong, and empty prompts each generate their own matched video.
