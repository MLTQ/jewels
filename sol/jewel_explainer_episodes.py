"""Plain-language, evidence-backed scripts for the six Jewel explainers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Shot:
    title: str
    narration: str
    caption: str
    visual: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Episode:
    number: int
    slug: str
    title: str
    subtitle: str
    sources: tuple[str, ...]
    shots: tuple[Shot, ...]
    theme: str = "dark"


EPISODES = (
    Episode(
        1,
        "prompt-to-program",
        "How a prompt becomes a video",
        "A direct tour from words to moving pixels",
        (
            "sol/prompt_trajectory_speaker.py",
            "sol/semantic_trajectory_realizer.py",
            "sol/prompt_video_runtime.py",
            "sol/results/jewel_casting_language_v0/PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md",
        ),
        (
            Shot(
                "Start with the whole idea",
                "Our prototype takes a short text prompt and a random seed. It turns them into a small plan, expands that plan into seventy-two thousand colored shapes called Jewels, and renders forty-nine frames. No finished video is supplied at generation time. The path really runs from words to a plan to a field of Jewels to moving pixels.",
                "prompt + seed → short plan → Jewel field → video",
                "pipeline",
                {"nodes": ["prompt + seed", "short plan", "Jewel field", "video"]},
            ),
            Shot(
                "What goes in—and what does not",
                "A system can appear creative while quietly copying most of its answer. We block that shortcut. Generation receives only the prompt and the declared random seed. It does not receive an input video, a fitted version of the target, a hidden target plan, or a saved target code. Some building blocks still come from training examples; that is an important limit, but the test video itself never enters the process.",
                "Only text and a seed enter generation; no target video is hidden inside.",
                "exclusion",
                {"allowed": ["prompt text", "random seed"], "forbidden": ["input video", "target field", "target plan", "hidden target code"]},
            ),
            Shot(
                "The prompt becomes a recipe",
                "Think of the first stage as a tiny recipe writer. It recognizes one of the three prompts the prototype currently knows, then uses the seed to choose two different training-owned ingredients. One ingredient will supply the moving subject and the other will supply the surroundings. The same prompt and seed always produce the same recipe, which makes every result reproducible.",
                "The prompt chooses the scene; the seed chooses two ingredients.",
                "tokens",
                {"tokens": ["scene", "subject", "setting"], "values": ["cartoon", "source 7", "source 11"]},
            ),
            Shot(
                "The recipe stays small",
                "The recipe does not list seventy-two thousand Jewels one by one. It names a scene, a subject ingredient, a setting ingredient, and a seed. Those names act like instructions for whole regions of the video, not individual pixels. This hierarchy matters: the short recipe keeps the subject consistent across time, while the many Jewels provide the visual detail.",
                "A few persistent instructions control many local details.",
                "program",
                {"rows": [("scene", "cartoon"), ("subject", "source 7"), ("setting", "source 11"), ("seed", "20260914")]},
            ),
            Shot(
                "Build a tube through time",
                "Imagine drawing the subject's path through a stack of video frames. The path forms a tube: a connected region that moves through space and time. We take the thirty-six thousand Jewels closest to that tube from the subject ingredient. We take thirty-six thousand far-away Jewels from the setting ingredient. The result has an exact size and a clear division of responsibility.",
                "36,000 subject Jewels + 36,000 setting Jewels = one field",
                "spacetime",
                {"mode": "rank-balanced", "foreground": 36000, "background": 36000},
            ),
            Shot(
                "Place the Jewels irregularly",
                "Each Jewel has a continuous position, a shape choice, a base-color choice, and a color-change choice. Continuous means its center can land anywhere inside the volume, not only at fixed grid points. A small random nudge prevents repeated positions from lining up. The output is therefore an irregular cloud of soft colored shapes rather than a blocky three-dimensional pixel grid.",
                "continuous center + shape + color + color change → one Jewel",
                "factorization",
                {"factors": [("center", "continuous", 0), ("shape", "1 of 1024", 1), ("color", "1 of 1024", 2), ("color change", "1 of 1024", 3)]},
            ),
            Shot(
                "What this proves",
                "The renderer turns the field into a normal video. In nine matched tests, the intended prompt was identified eight times, and correct prompts beat wrong or empty prompts in every comparison. That is real evidence that text controls the generated field. It is not yet a general text-to-video model: the vocabulary contains only three prompts and eighteen training-owned ingredients. We have proved the route, not the scale.",
                "The route works; the vocabulary is still deliberately tiny.",
                "playback",
                {"clip": "ballerina", "metrics": ["8 of 9 prompt matches", "9 of 9 control wins", "no grid locking"]},
            ),
        ),
    ),
    Episode(
        2,
        "jewel-geometry",
        "What is a spacetime Gaussian Jewel?",
        "A soft shape that exists across position and time",
        (
            "stprim/core/params.py",
            "stprim/prior/featurize.py",
            "stprim/models/render.py",
            "stprim/models/tiled_support.py",
            "sol/generate_jewel_isolation_asset.py",
            "sol/results/jewel_explainer_series_v1/assets/actual_jewel_isolation.json",
        ),
        (
            Shot(
                "A soft blob through time",
                "Picture a video as a stack of transparent sheets, one sheet per frame. A Jewel is a soft three-dimensional blob placed inside that stack. Its three directions are left-to-right, up-and-down, and time. Mathematicians call its smooth bell-shaped fade a Gaussian: it is strongest at the center and gradually fades away. A visible frame is simply one slice through the blob.",
                "A Jewel is a soft Gaussian blob in left-right, up-down, and time.",
                "spacetime",
                {"mode": "slice", "foreground": 0, "background": 0},
            ),
            Shot(
                "Meet four actual Jewels",
                "This is a real fitted video of a singer at a microphone, built from six thousand four hundred seventy-one Jewels. We mark four strong, moving Jewels, fade away every other contribution, and follow those same four through all sixty-four frames. Their centers move and their visible cross-sections change as time passes. We brighten the isolated view equally so the small contributions remain easy to see; their fitted positions, shapes, and colors are unchanged.",
                "Full fitted video, then four actual Jewels, then full fitted video.",
                "jewel-isolation",
                {"clip": "actual-jewels", "count": 4},
            ),
            Shot(
                "Five kinds of information",
                "A Jewel stores five things. Its center says where and when it lives. Its shape says how wide it is and which way it tilts. Its base color says what it looks like at the center. A color-change table says how that color varies nearby. Finally, a strength value says how much it contributes. The technical name for the shape table is covariance. Together these use twenty-two numbers.",
                "center | shape | base color | nearby color change | strength",
                "feature-vector",
                {"segments": [("center", 3, 0), ("shape", 6, 1), ("color", 3, 2), ("color change", 9, 3), ("strength", 1, 4)]},
            ),
            Shot(
                "Tilt creates motion",
                "Imagine pushing a cucumber through that stack of frame sheets at an angle. Each sheet cuts the cucumber at a slightly different horizontal position, so the slice appears to move. A tilted Jewel works the same way. If it stretches mostly through time, it persists. If it tilts across both space and time, its visible position moves from frame to frame. Motion is built into the shape itself.",
                "A tilted spacetime shape becomes motion when sliced into frames.",
                "equation",
                {"equation": "shape = widths + tilt through (u, v, time)", "diagram": "ellipsoid"},
            ),
            Shot(
                "Color can vary nearby",
                "The base color describes the exact center. A small three-by-three table describes how red, green, and blue change when we move left, up, or forward in time. That table is called a color Jacobian. We will use the full name instead of unexplained shorthand. Think of it as three tiny color slopes, including a slope through time.",
                "color Jacobian = how red, green, and blue change across space and time",
                "equation",
                {"equation": "local color = base color + color-change table × offset", "diagram": "gradient"},
            ),
            Shot(
                "Many Jewels paint one pixel",
                "At each pixel, every nearby Jewel paints a small amount of color. Strong, close Jewels contribute more; weak or distant Jewels contribute less. We simply add those contributions to a learned background color. This is called an additive renderer because the contributions are added, like overlapping pools of light. No Jewel must win exclusive ownership of the pixel.",
                "background + all nearby Jewel contributions = the final pixel",
                "equation",
                {"equation": "pixel = background + sum of nearby Jewel colors", "diagram": "sum"},
            ),
            Shot(
                "Check only where a Jewel matters",
                "A Jewel fades forever in theory, but after five of its own widths the remaining effect is tiny. We call the region where it can matter its support. The renderer divides the volume into boxes so it can quickly find every Jewel whose support reaches a pixel. This is a filing system, not a placement grid: Jewel centers remain irregular, and long tilted shapes are not accidentally missed.",
                "A box index speeds up lookup without snapping Jewels to a grid.",
                "support",
                {"sigma": 5.0, "boundary": "0.0000037", "neighbors": 27},
            ),
        ),
        theme="light",
    ),
    Episode(
        3,
        "native-vocabulary",
        "Giving the model a Jewel vocabulary",
        "Reusable words for shape and appearance",
        (
            "stprim/prior/featurize.py",
            "sol/factorized_jewel_casting_language.py",
            "sol/prompt_jewel_caster.py",
            "sol/results/jewel_casting_language_v0/hierarchical_v1/gate0f_individual/report.json",
        ),
        (
            Shot(
                "Why make tokens at all?",
                "A language model works with reusable choices called tokens: words or word-like symbols from a fixed vocabulary. We want similar Jewel shapes and colors to reuse the same tokens across many videos. This is not an attempt to compress a finished video. The goal is to give a future generator a stable set of physical words it can speak into an empty spacetime volume.",
                "Tokens are reusable physical choices, not a video compression format.",
                "comparison",
                {"left": "raw fitted numbers", "right": "reusable Jewel words", "left_value": "one-off", "right_value": "shared"},
            ),
            Shot(
                "One shape can have many spellings",
                "The same tilted ellipsoid can be written with several different rotation-number combinations. The picture does not change, but the numbers do. A learner would mistake those spellings for different shapes and waste vocabulary space. Before creating tokens, we convert every Jewel to one consistent numerical spelling. In mathematics this removes a gauge ambiguity; in plain language, it removes meaningless aliases.",
                "First give every visible shape one consistent numerical spelling.",
                "equation",
                {"equation": "different rotation numbers → the same visible ellipsoid", "diagram": "gauge"},
            ),
            Shot(
                "A stable six-number shape",
                "We store each shape as a symmetric three-by-three table, which needs six unique numbers. We also take a matrix logarithm, a standard transformation that makes large and small scales easier to compare. The important result is simple: identical visible shapes now receive identical stored features. The later renderer can convert that stable spelling back into widths and tilt.",
                "A unique six-number shape description replaces ambiguous rotations.",
                "feature-vector",
                {"segments": [("center", 3, 0), ("shape", 6, 1), ("color", 3, 2), ("color change", 9, 3), ("strength", 1, 4)]},
            ),
            Shot(
                "Separate the physical choices",
                "Our first attempt bundled layout, shape, color, and color change into one enormous choice. That was like printing every possible sentence as a single dictionary entry. It failed. The stronger design gives each role its own list of prototypes. A prototype is simply a learned representative example. The model can then combine a familiar shape with a familiar color in a new way.",
                "Choose layout, shape, color, and color change separately.",
                "factorization",
                {"factors": [("layout", "1024 choices", 0), ("shape", "1024 choices", 1), ("color", "1024 choices", 2), ("color change", "1024 choices", 3)]},
            ),
            Shot(
                "Mix-and-match creates range",
                "Four lists of one thousand twenty-four choices can describe far more combinations than one list of the same size. We do not build a table containing every combination; we compose the roles when needed. For an individual Jewel, position remains continuous while shape, color, and color change come from their learned vocabularies. This gives the speaker reusable parts without forcing every result to copy a stored whole.",
                "A few reusable lists can be mixed into many Jewel combinations.",
                "tokens",
                {"tokens": ["shape", "color", "color change"], "values": ["choice 418", "choice 77", "choice 903"]},
            ),
            Shot(
                "A map is not a parking space",
                "The learner uses coarse boxes to ask which tokens are common in each neighborhood. Those boxes are addresses for routing information, like postal codes. They are not the final Jewel positions, just as a postal code is not a chair inside a house. The model still emits a continuous center that can land anywhere. Discrete routing and irregular geometry can safely coexist.",
                "A routing box selects context; it does not snap the Jewel's center.",
                "spacetime",
                {"mode": "continuous", "foreground": 0, "background": 0},
            ),
            Shot(
                "The vocabulary keeps the signal",
                "We tested whether replacing continuous features with vocabulary choices destroyed the field. The reconstructed images reached about twenty-two point nine decibels of peak signal-to-noise ratio, a standard image-similarity measure where higher is better. Spacetime tilt was preserved, and no centers locked to the routing grid. This does not prove promptable generation. It shows the physical vocabulary is usable enough for the next level.",
                "Useful image fidelity, preserved motion tilt, and zero center locking.",
                "metrics",
                {"bars": [("image fidelity", 22.8657, 25.0, 0), ("tilt retained", 1.0415, 1.2, 1), ("grid locking", 0.0, 1.0, 4)]},
            ),
        ),
    ),
    Episode(
        4,
        "coherence-trajectories",
        "Why motion needs a persistent owner",
        "The experiments that turned texture into subjects",
        (
            "sol/block_token_jewel_speaker.py",
            "sol/coherent_source_realizer.py",
            "sol/semantic_trajectory_realizer.py",
            "sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md",
        ),
        (
            Shot(
                "Good pieces can make a bad whole",
                "Our first generator chose each local Jewel from the prompt and its nearby position. Individual patches often had plausible colors and textures, yet the whole video had no recognizable subject. It was the visual version of assembling a jigsaw from individually convincing pieces taken from different boxes. Local correctness does not automatically create one object that survives across time.",
                "Plausible local pieces do not guarantee one coherent subject.",
                "comparison",
                {"left": "independent pieces", "right": "persistent owner", "left_value": "texture", "right_value": "subject"},
            ),
            Shot(
                "Larger patches still lacked identity",
                "We next predicted small blocks of Jewels instead of single ones. The model became better at predicting which block belonged in each neighborhood, but the rendered result was still texture. That was useful evidence: the local statistics were improving, while identity was still missing. Something had to tell distant regions and later frames that they belonged to the same subject.",
                "Better local prediction still did not create a persistent subject.",
                "metrics",
                {"bars": [("global error", 6.1234, 7.0, 4), ("local error", 5.1680, 7.0, 1), ("recognizable", 0.0, 3.0, 5)]},
            ),
            Shot(
                "One owner brought subjects back",
                "As a diagnostic test, we assigned one complete training-owned program to the entire spacetime window. The physical Jewel vocabulary did not change, but ballerina, dog, and welder subjects became recognizable again. This test was retrieval, not new generation. Its value was causal: it showed that persistent ownership across blocks and frames was the missing ingredient.",
                "Same Jewels, one persistent owner: recognizable subjects return.",
                "evidence",
                {"asset": "coherent", "label": "persistent-owner diagnostic"},
            ),
            Shot(
                "Mix two sources to test composition",
                "A complete training program could merely replay one example. To test that, we built a new field from two different programs. One supplied Jewels near the moving subject tube; the other supplied the surroundings. When we deliberately swapped in a wrong subject source, the visible subject changed. The result was not a replay of either complete source, and the tube truly controlled subject identity.",
                "A moving subject and a separate setting form a new field.",
                "spacetime",
                {"mode": "donors", "foreground": 36000, "background": 36000},
            ),
            Shot(
                "Find where the subject differs",
                "The subject path comes from a saliency map. Saliency simply means a map of what stands out. For each time step, we compare the chosen scene with the other scenes and find the region where their learned descriptions differ most. A small smoothing pass prevents the path from jumping. This gives us a class-related trajectory without looking at the target test video.",
                "A difference map finds the subject region, then smoothing connects the path.",
                "equation",
                {"equation": "subject path = center of the scene difference map over time", "diagram": "path"},
            ),
            Shot(
                "Choose by rank, not radius",
                "Different fields can pack Jewels at different densities. A fixed tube width might collect too many from one source and too few from another. Instead, we sort by distance from the path. We take the closest thirty-six thousand subject Jewels and the farthest thirty-six thousand setting Jewels. Ranking guarantees the requested count without duplicates or target-specific tuning.",
                "Sort by tube distance to get an exact, balanced field.",
                "equation",
                {"equation": "closest 36,000 subject + farthest 36,000 setting", "diagram": "rank"},
            ),
            Shot(
                "Change the cause, not just the label",
                "A pretty sample is not enough. For every prompt and seed, we generate three fields: one from the correct prompt, one from a deliberately wrong prompt, and one with no prompt meaning at all. These are causal controls: we change the input that should cause the result, then check whether the generated field changes in the intended direction. The video itself is regenerated for every condition.",
                "Correct, wrong, and empty prompts each generate their own matched video.",
                "evidence",
                {"asset": "proof-sheet", "label": "matched causal-control frames"},
            ),
        ),
    ),
    Episode(
        5,
        "evidence-gates",
        "How we decide whether it really works",
        "Tests designed to resist wishful thinking",
        (
            "sol/audit_prompt_trajectory_speaker.py",
            "sol/train_learned_trajectory_speaker.py",
            "sol/audit_learned_trajectory_speaker.py",
            "sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md",
        ),
        (
            Shot(
                "Write the rules before the run",
                "Before generating results, we freeze the prompts, seeds, renderer, sampled frames, scoring method, and pass thresholds. This is called preregistration: deciding what counts as success before seeing which samples look good. We test three prompts, three seeds, and three prompt conditions. An automatic text-image scorer compares three frames from each video with the intended text.",
                "Freeze the test and pass line before looking at the results.",
                "pipeline",
                {"nodes": ["3 video frames", "text-image scorer", "video score", "pass or fail"]},
            ),
            Shot(
                "Ask two different questions",
                "The first test asks whether a generated video matches its intended prompt better than the other two prompt labels. The second asks whether correct text produces a better video than wrong or empty text, while keeping the intended meaning fixed for scoring. The first checks recognition. The second checks causal control. We also count individual wins so a good average cannot hide repeated failures.",
                "Recognition ranks labels; causal testing compares separately generated videos.",
                "comparison",
                {"left": "one video, three labels", "right": "one meaning, three videos", "left_value": "recognition", "right_value": "causal control"},
            ),
            Shot(
                "The exact recipe passes",
                "Across nine videos made by the exact recipe writer, eight were matched to the intended prompt. Correctly prompted videos beat wrong-prompt videos in all nine comparisons and beat empty-prompt videos in all nine. The average margins cleared the thresholds we set in advance. The field-size, rendering, and irregular-position checks also passed.",
                "8 of 9 recognized · 9 of 9 beat wrong prompts · 9 of 9 beat empty prompts",
                "metrics",
                {"bars": [("recognized", 8, 9, 0), ("beat wrong", 9, 9, 1), ("beat empty", 9, 9, 2), ("grid locking", 0, 1, 4)]},
            ),
            Shot(
                "Then replace rules with learning",
                "The exact writer uses hand-built prompt lookup. Our next test replaced it with a small learned network containing about five hundred forty-one thousand adjustable numbers. The network reads a text embedding, which is a numerical summary of the sentence. It predicts the scene first, then the subject source, then the setting source. Only choosing the exact same source twice is forbidden; scene-consistent choices must be learned.",
                "text summary → scene choice → subject choice → setting choice",
                "pipeline",
                {"nodes": ["text summary", "scene", "subject", "setting"]},
            ),
            Shot(
                "Test unfamiliar wording and pairings",
                "We hold back both a new paraphrase and specific subject-setting combinations. Held out means the learner does not see those examples during training. At evaluation, it must understand different wording and choose a source pair it did not memorize as a pair. Empty-text examples teach the network what no prompt looks like. This makes the test stricter than repeating the training phrases.",
                "Evaluation changes both the wording and the source pairing.",
                "tokens",
                {"tokens": ["training wording", "new wording", "new pairing"], "values": ["3 per class", "1 per class", "18 programs"]},
            ),
            Shot(
                "More training reached a plateau",
                "We continued training and measured the held-out error every one hundred steps. The best result appeared at step one hundred. Ten later measurements, through step eleven hundred, did not improve it. That flat stretch is a plateau: more of the same training was no longer helping. The network separated correct wording from wrong wording, but it also began fitting the tiny training set too closely.",
                "Best at step 100; no improvement through step 1,100.",
                "curve",
                {"series": [("correct", [2.20, 2.32, 2.47, 2.58, 2.66, 2.73], 1), ("wrong", [4.39, 4.55, 4.72, 4.83, 4.92, 5.02], 4), ("empty", [2.54, 2.61, 2.68, 2.73, 2.78, 2.82], 0)]},
            ),
            Shot(
                "Keep the near-pass honest",
                "The learned writer chose the correct scene for all nine held-out prompts and usually beat the causal controls. But only four of nine raw videos were matched to the right prompt, below our required six. So the learned gate fails. The useful conclusion is narrower: the network learned the tiny recipe syntax and responded to text, but the data and vocabulary were too small for robust prompt-to-video meaning.",
                "The learned writer responds to text, but strict recognition is only 4 of 9.",
                "evidence",
                {"asset": "evidence", "label": "exact pass and learned near-pass"},
            ),
        ),
    ),
    Episode(
        6,
        "scaling-to-t2v",
        "What would turn this into text-to-video?",
        "The next data, model, and evaluation steps",
        (
            "sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md",
            "sol/results/jewel_casting_language_v0/PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md",
            ".beads/issues.jsonl",
        ),
        (
            Shot(
                "What we have—and what is missing",
                "We now have a connected path from text to a short persistent plan, from that plan to continuous Jewels, and from Jewels to video. We also know that persistent trajectories can preserve recognizable subjects. The missing piece is breadth. Today's subject and setting instructions still point back to specific training fields. A real model needs reusable ideas such as dog, walking, beach, and camera pan that can combine beyond their source videos.",
                "The full route exists; reusable, broad concepts are the missing piece.",
                "pipeline",
                {"nodes": ["text", "persistent plan", "continuous Jewels", "video"]},
            ),
            Shot(
                "Do not scale source labels",
                "Today, an instruction such as subject seven ultimately means material taken from one fitted training field. Adding thousands of source numbers would only build a larger filing cabinet of examples. Instead, we must discover patterns repeated across many videos and give those patterns shared names. A walking-dog instruction should be learned from many dogs and many walks, not tied to one clip.",
                "Replace source labels with concepts learned across many videos.",
                "comparison",
                {"left": "source number 7", "right": "shared concept 7", "left_value": "one field", "right_value": "many fields"},
            ),
            Shot(
                "The next decisive experiment",
                "Fit Jewel fields for at least one hundred videos covering ten to twenty prompts with useful combinations. Then learn sixty-four to two hundred fifty-six recurring subject and setting patterns. Each pattern should own a connected path through time while reusing the existing shape and color vocabularies. Finally, reject any generated result that copies too much from one source. Passing this test would show genuine reusable building blocks.",
                "100+ fields → recurring paths → reusable concepts → no-copy test",
                "future",
                {"stages": ["100+ fields", "find recurring paths", "learn shared concepts", "run no-copy test"]},
            ),
            Shot(
                "Speak plans, not seventy-two thousand rows",
                "A future language model should not recite every Jewel one at a time. It should speak a hierarchy, like a film crew's plan: scene and camera first, then persistent objects, then their paths, then local appearance details. Smaller specialist modules can expand that plan into exact Jewel positions and shapes. This lets the language model spend its capacity on meaning and long-range consistency rather than repetitive rendering instructions.",
                "scene → objects → paths → local detail → continuous Jewels",
                "pipeline",
                {"nodes": ["scene + camera", "objects", "paths", "Jewel detail"]},
            ),
            Shot(
                "Carry motion into the next window",
                "Long video will be generated in neighboring time windows. The model must carry state from one window to the next: object identity, position, direction, speed, and style. Speed and direction are derivatives, meaning measurements of how position is changing. Predicting those changes worked better than predicting every next position independently. It is the difference between continuing a thrown ball's flight and guessing a fresh location every frame.",
                "Carry identity, position, direction, speed, and style across windows.",
                "equation",
                {"equation": "next position = current position + speed × elapsed time", "diagram": "derivative"},
            ),
            Shot(
                "Make the next test hard to fake",
                "Training and evaluation should split whole combinations, such as an object doing an action, instead of randomly splitting similar videos. Correct text must beat swapped-object and swapped-action controls. One prompt should produce several recognizable but genuinely different samples. A second time window must continue from the model's own carried state, without peeking at a target video. These checks separate composition from memorization.",
                "Hold out combinations, demand diversity, and continue without target help.",
                "exclusion",
                {"allowed": ["prompt", "seed", "generated state"], "forbidden": ["source label", "target video", "teacher path", "seen combination"]},
            ),
            Shot(
                "The honest compute pitch",
                "More compute matters only when paired with the right data and structure. More varied videos reveal which objects, actions, cameras, and paths repeat. More training can turn those repetitions into shared concepts and learn how prompts combine them. More rendering compute can improve local detail. The pitch is not that size will magically repair random splats. It is that the full route now works at toy scale, and the next uncertainty can be attacked directly with data and compute.",
                "More varied data + reusable concepts + a larger plan model = the next proof.",
                "future",
                {"stages": ["varied data", "shared concepts", "larger plan model", "longer carry", "open prompts"]},
            ),
        ),
    ),
)


def episode_by_number(number: int) -> Episode:
    try:
        return next(episode for episode in EPISODES if episode.number == number)
    except StopIteration as error:
        raise ValueError(f"unknown episode {number}") from error
