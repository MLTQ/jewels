# Episode 6: What would turn this into text-to-video?

The next data, model, and evaluation steps

## Claim sources

- `sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md`
- `sol/results/jewel_casting_language_v0/PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md`
- `.beads/issues.jsonl`

## 1. What we have—and what is missing

We now have a connected path from text to a short persistent plan, from that plan to continuous Jewels, and from Jewels to video. We also know that persistent trajectories can preserve recognizable subjects. The missing piece is breadth. Today's subject and setting instructions still point back to specific training fields. A real model needs reusable ideas such as dog, walking, beach, and camera pan that can combine beyond their source videos.

**On screen:** The full route exists; reusable, broad concepts are the missing piece.

## 2. Do not scale source labels

Today, an instruction such as subject seven ultimately means material taken from one fitted training field. Adding thousands of source numbers would only build a larger filing cabinet of examples. Instead, we must discover patterns repeated across many videos and give those patterns shared names. A walking-dog instruction should be learned from many dogs and many walks, not tied to one clip.

**On screen:** Replace source labels with concepts learned across many videos.

## 3. The next decisive experiment

Fit Jewel fields for at least one hundred videos covering ten to twenty prompts with useful combinations. Then learn sixty-four to two hundred fifty-six recurring subject and setting patterns. Each pattern should own a connected path through time while reusing the existing shape and color vocabularies. Finally, reject any generated result that copies too much from one source. Passing this test would show genuine reusable building blocks.

**On screen:** 100+ fields → recurring paths → reusable concepts → no-copy test

## 4. Speak plans, not seventy-two thousand rows

A future language model should not recite every Jewel one at a time. It should speak a hierarchy, like a film crew's plan: scene and camera first, then persistent objects, then their paths, then local appearance details. Smaller specialist modules can expand that plan into exact Jewel positions and shapes. This lets the language model spend its capacity on meaning and long-range consistency rather than repetitive rendering instructions.

**On screen:** scene → objects → paths → local detail → continuous Jewels

## 5. Carry motion into the next window

Long video will be generated in neighboring time windows. The model must carry state from one window to the next: object identity, position, direction, speed, and style. Speed and direction are derivatives, meaning measurements of how position is changing. Predicting those changes worked better than predicting every next position independently. It is the difference between continuing a thrown ball's flight and guessing a fresh location every frame.

**On screen:** Carry identity, position, direction, speed, and style across windows.

## 6. Make the next test hard to fake

Training and evaluation should split whole combinations, such as an object doing an action, instead of randomly splitting similar videos. Correct text must beat swapped-object and swapped-action controls. One prompt should produce several recognizable but genuinely different samples. A second time window must continue from the model's own carried state, without peeking at a target video. These checks separate composition from memorization.

**On screen:** Hold out combinations, demand diversity, and continue without target help.

## 7. The honest compute pitch

More compute matters only when paired with the right data and structure. More varied videos reveal which objects, actions, cameras, and paths repeat. More training can turn those repetitions into shared concepts and learn how prompts combine them. More rendering compute can improve local detail. The pitch is not that size will magically repair random splats. It is that the full route now works at toy scale, and the next uncertainty can be attacked directly with data and compute.

**On screen:** More varied data + reusable concepts + a larger plan model = the next proof.
