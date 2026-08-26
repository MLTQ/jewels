# Episode 4: Why coherence needs persistent ownership

The failure ladder that led to trajectory programs

## Claim sources

- `sol/block_token_jewel_speaker.py`
- `sol/coherent_source_realizer.py`
- `sol/semantic_trajectory_realizer.py`
- `sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md`

## 1. Independent Jewels make texture

The naïve generator predicts each Jewel mark from global text and local coordinates. It can learn marginal colors, covariance frequencies, and density. What it does not own is a persistent object. Independent local choices are statistically plausible but mutually inconsistent: one region votes for fur, another for metal, and successive time slabs disagree about identity. The result is structured texture rather than a recognizable actor performing an action.

**On screen:** Correct local marginals do not imply a coherent global subject.

## 2. Local block phrases were better—but insufficient

Addressing a sixteen-by-sixteen-by-eight spacetime grid and predicting one of one-thousand-and-twenty-four local block phrases lowers direct token negative log likelihood to five-point-one-six-eight, fifteen-point-six percent better than the global posterior. Yet qualitative renders remain texture. That experiment is vital because it separates two hypotheses. Local statistical modeling works. Recognizable composition still fails because no variable binds those local phrases into one object trajectory.

**On screen:** Addressed phrases: NLL 5.168, 15.60% better—still no subject.

## 3. One coherent owner restores the subject

The next oracle chooses one complete training-owned program for the entire spacetime window. Nothing about the physical vocabulary changes. The same continuous centroids and one-thousand-and-twenty-four-way active tokens suddenly render recognizable ballerina, dog, and welder examples. This is retrieval-only and therefore not the final generator. But causally, it identifies persistent cross-block ownership as the missing variable.

**On screen:** Same physical Jewels + one persistent owner → all three subjects return.

## 4. Two donors rule out whole-video retrieval

A complete-source owner could be dismissed as nearest-neighbor playback. So the next experiment composes two distinct training programs. A moving foreground tube selects material from one donor; everything outside is supplied by another. Wrong-object controls deliberately take the foreground donor from a different semantic scene while preserving the path. The visible subject swaps. The field is therefore a new composition, and the tube has causal semantic ownership.

**On screen:** Distinct foreground and background donors form a new field.

## 5. The semantic path comes from training-only saliency

For each addressed block, the realizer compares the scene's mean normalized descriptor with the mean of all other scenes and squares the difference. A centered spatial prior suppresses unstable borders. Summing saliency over u and v yields one center per time slab, then a one-two-one temporal filter smooths the path. The path therefore follows where a class differs from alternatives, not where two arbitrary source fields happen to disagree.

**On screen:** path(t) = center of scene-vs-other descriptor saliency, smoothed [1,2,1]

## 6. Rank balance fixes density mismatch

A single geometric radius can select unequal totals because different valid fields have different local densities. Instead, Gate two-b-zero sorts foreground Jewels by increasing squared tube distance and takes the closest half-budget. It sorts background Jewels in the opposite direction and takes the farthest half-budget. Top-k selection makes the emitted count exact, forces fifty-fifty ownership, and requires no adjustment duplicates or target-derived threshold.

**On screen:** top-k nearest foreground + top-k farthest background; no density assumption.

## 7. Causal controls, not one pretty sample

Every intended prompt is rendered under three seeds. Under the same declared seed, a cyclic-shuffled prompt compiles the next semantic scene, while a null prompt derives scene from seed alone. Correct programs must beat those generated controls, not merely score against alternative text on an unchanged render. This distinction matters: it tests whether changing text causes the generated field to change in the intended direction.

**On screen:** Correct, cyclic-shuffled, and null prompts generate matched causal controls.
