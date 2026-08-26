"""Evidence-backed narration and shot specifications for six Jewel explainers."""

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


EPISODES = (
    Episode(
        1,
        "prompt-to-program",
        "From words to a spacetime program",
        "The exact inference path, without an input video",
        (
            "sol/prompt_trajectory_speaker.py",
            "sol/semantic_trajectory_realizer.py",
            "sol/prompt_video_runtime.py",
            "sol/results/jewel_casting_language_v0/PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md",
        ),
        (
            Shot(
                "The claim, stated precisely",
                "The current result is a bounded existence proof, not a general text-to-video model. Given one of three registered prompts and an integer seed, the system emits a finite native program, expands that program into exactly seventy-two thousand irregular Jewels, and renders forty-nine video frames. The important point is causal direction. At inference, information flows from text to program to a continuous spacetime field to pixels. It does not flow from a target video backward into an encoding.",
                "prompt + seed → typed program → 72,000 Jewels → 49 rendered frames",
                "pipeline",
                {"nodes": ["prompt + seed", "typed program", "Jewel field", "video"]},
            ),
            Shot(
                "What the generator is forbidden to see",
                "A reconstruction system can look generative while secretly receiving most of the answer. So Gate two-b-zero explicitly removes four channels: no input video, no fitted target field, no target-derived block program, and no held-out latent. The only semantic input is the exact prompt string. The only stochastic input is the declared integer seed. Training-owned source tokens remain in the vocabulary; that is a limitation we will return to, but no test target participates in compilation or casting.",
                "No video, target field, target program, or hidden latent enters inference.",
                "exclusion",
                {"allowed": ["prompt text", "integer seed"], "forbidden": ["input video", "target field", "block program", "held-out latent"]},
            ),
            Shot(
                "The exact compiler",
                "The exact speaker normalizes whitespace, looks up the prompt's dense scene index, and then creates a deterministic random generator from the declared seed plus one-thousand-and-nine times the scene. A seeded permutation of the six source tokens owned by that scene chooses two distinct entries. The first becomes the foreground trajectory token; the second becomes the background token. Therefore prompt plus seed uniquely determines the program, while changing the seed changes the donor pair without consulting any pixels.",
                "scene = lookup(prompt); order = randperm(seed + 1009 * scene)",
                "tokens",
                {"tokens": ["scene", "foreground", "background"], "values": ["1", "7", "11"]},
            ),
            Shot(
                "One three-token utterance",
                "A compiled utterance contains a semantic scene token, a foreground token, a background token, the seed, and an audit condition. Scene is not a pixel class. It selects a persistent semantic path through the entire spacetime window. Foreground and background are not individual Gaussians either. They name coherent, source-backed macro programs containing tens of thousands of physical Jewel tokens. This hierarchy is the central design decision: the small program owns global coherence; local Jewels own rendering detail.",
                "scene / foreground / background are persistent program tokens—not pixels.",
                "program",
                {"rows": [("scene", "cartoon / token 1"), ("foreground", "source token 7"), ("background", "source token 11"), ("seed", "20260914")]},
            ),
            Shot(
                "Casting through a moving tube",
                "For every time slab, the realizer stores a two-dimensional semantic path. It computes each donor Jewel's squared distance from that path. The foreground contribution is the thirty-six thousand closest Jewels from the foreground donor. The background contribution is the thirty-six thousand farthest Jewels from the background donor. This rank-balanced construction is deliberately count-exact. It avoids assuming that two valid fields have the same density inside one fixed radius, while forcing both donors to own exactly half of the final field.",
                "closest 36,000 foreground + farthest 36,000 background = 72,000",
                "spacetime",
                {"mode": "rank-balanced", "foreground": 36000, "background": 36000},
            ),
            Shot(
                "Physical tokens become Jewels",
                "Each selected row carries a continuous centroid and three active physical token identifiers: covariance, surface color, and color gradient. Every active vocabulary contains one-thousand-and-twenty-four prototypes. Decoding substitutes the selected prototype values into the canonical twenty-two-dimensional feature layout while leaving the centroid continuous. A small Gaussian jitter is applied and clamped strictly inside normalized volume bounds. The result is an irregular set, not a Cartesian output grid.",
                "continuous μ + covariance token + surface token + gradient token → one Jewel",
                "factorization",
                {"factors": [("mu", "continuous", 0), ("Sigma", "K=1024", 1), ("RGB", "K=1024", 2), ("JRGB", "K=1024", 3)]},
            ),
            Shot(
                "A bounded but genuine generation path",
                "The support-correct renderer evaluates that field on every requested frame grid and encodes the result as an H.264 video. The proof passes because text controls which coherent programs are cast: across nine exact programs, intended prompt retrieval is eight of nine, and correct prompt programs beat both cyclic-shuffled and null generations in all nine matched cases. But the vocabulary is still three prompts and eighteen source-backed macro tokens. This proves the mechanism, not the final scale.",
                "The path is genuinely generative; the vocabulary is deliberately tiny.",
                "playback",
                {"clip": "ballerina", "metrics": ["8/9 prompt retrieval", "9/9 shuffled wins", "0% grid locking"]},
            ),
        ),
    ),
    Episode(
        2,
        "jewel-geometry",
        "A Jewel is a Gaussian in spacetime",
        "Geometry, appearance, and the exact additive renderer",
        (
            "stprim/core/params.py",
            "stprim/prior/featurize.py",
            "stprim/models/render.py",
            "stprim/models/tiled_support.py",
        ),
        (
            Shot(
                "Three coordinates, not four",
                "A Jewel lives in normalized u, v, t coordinates: horizontal image position, vertical image position, and time. This is a three-dimensional spacetime volume. RGB is attached appearance, not a fourth Gaussian axis. To render frame t-zero, we slice the volume with a plane at that time and evaluate every pixel coordinate on the plane. Time distortion comes from an anisotropic covariance whose principal axes may tilt between space and time.",
                "Jewel geometry lives in (u, v, t); a video frame is a time slice.",
                "spacetime",
                {"mode": "slice", "foreground": 0, "background": 0},
            ),
            Shot(
                "The twenty-two canonical features",
                "The generative boundary represents each Jewel as twenty-two numbers. Three numbers store the centroid mu. Six store the upper triangle of the symmetric log-covariance. Three store constant RGB. Nine store the full three-by-three world-frame color Jacobian. One stores the weight logit. This layout is gauge-free: appearance gradients are expressed in world coordinates, and covariance is stored as a unique symmetric matrix rather than a rotation convention.",
                "μ(3) | log Σ upper triangle(6) | RGB(3) | color Jacobian(9) | logit weight(1)",
                "feature-vector",
                {"segments": [("μ", 3, 0), ("log Σ", 6, 1), ("RGB", 3, 2), ("JRGB", 9, 3), ("logit w", 1, 4)]},
            ),
            Shot(
                "Covariance controls time distortion",
                "Internally the renderer factorizes covariance into a proper rotation R and positive principal scales S. A Jewel elongated along u paints a horizontal region. One elongated along t persists across frames. A principal axis tilted jointly through u and t moves across the image as time advances. That mixed spacetime tilt is not post-hoc optical flow; it is literally encoded in the orientation of the Gaussian ellipsoid.",
                "Sigma = R diag(s^2) R^T; tilted eigenvectors couple position and time.",
                "equation",
                {"equation": "Sigma = R * diag(s_u^2, s_v^2, s_t^2) * R^T", "diagram": "ellipsoid"},
            ),
            Shot(
                "Mahalanobis distance",
                "For a query point x, displacement d equals x minus mu. The renderer rotates that displacement into the Jewel's principal frame with R-transpose, divides componentwise by scale, and sums the squares. This yields q, the squared Mahalanobis distance. Computing y equals S inverse R-transpose d avoids materializing an inverse covariance for every pixel-Jewel pair. The Gaussian contribution decays as exponential minus one-half q.",
                "d = x - mu; y = S^-1 R^T d; q = ||y||^2",
                "equation",
                {"equation": "q_i(x) = (x-mu_i)^T Sigma_i^-1 (x-mu_i) = ||S_i^-1 R_i^T (x-mu_i)||^2", "diagram": "distance"},
            ),
            Shot(
                "Appearance is locally linear",
                "A constant-color Gaussian would need many small primitives to express a smooth color ramp. Instead, the default P-one appearance model stores a three-by-three Jacobian. Local color is base color plus that Jacobian times displacement from the centroid. One row describes how red changes with u, v, and t; the next rows do the same for green and blue. Temporal entries can therefore change a Jewel's color as a frame slice moves through it.",
                "c_i(x) = c_i^0 + J_i(x - mu_i)",
                "equation",
                {"equation": "c_i(x) = RGB_i + JRGB_i * (x - mu_i)", "diagram": "gradient"},
            ),
            Shot(
                "The field is additive",
                "The final pixel is the learned constant background plus a sum of every supported Jewel contribution. Each weight is sigmoid of its stored logit, multiplied by the Gaussian exponential, multiplied by local color. There is no softmax normalization forcing Jewels to compete for ownership. An earlier soft-Voronoi model was implemented and steelmanned, but lost both reconstruction and canonicality, so the evidence path is explicitly additive.",
                "value(x) = background + sum_i exp(-q_i/2) * sigmoid(a_i) * c_i(x)",
                "equation",
                {"equation": "I(x) = b + sum_i exp(-q_i(x)/2) sigmoid(a_i) c_i(x)", "diagram": "sum"},
            ),
            Shot(
                "Support-complete rendering",
                "Nearest-center culling is not mathematically safe for anisotropic splats. A long tilted ellipsoid may cover a pixel even when many narrow Gaussians have closer centers. The evidence renderer truncates at five standard deviations, builds an exact world-axis bounding box for each support ellipsoid, assigns every primitive to one multilevel tile, queries twenty-seven neighboring cells, and applies the true Mahalanobis test. Beyond five sigma, boundary weight is approximately three-point-seven times ten to the minus six.",
                "Five-sigma tiled support is complete; Euclidean center KNN is not.",
                "support",
                {"sigma": 5.0, "boundary": "3.7e-6", "neighbors": 27},
            ),
        ),
    ),
    Episode(
        3,
        "native-vocabulary",
        "Turning Jewels into a native language",
        "Gauge-free features, factor tokens, and continuous positions",
        (
            "stprim/prior/featurize.py",
            "sol/factorized_jewel_casting_language.py",
            "sol/prompt_jewel_caster.py",
            "sol/results/jewel_casting_language_v0/hierarchical_v1/gate0f_individual/report.json",
        ),
        (
            Shot(
                "Why tokenization is nontrivial",
                "A language needs recurring symbols, but fitted Gaussian parameters are not naturally unique. Two optimization runs can describe the same visual field with different row order, slightly shifted centers, and different rotation parameterizations. Quantizing raw parameter tuples would spend vocabulary capacity on accidental coordinate choices. The first job is therefore not compression. It is to define a stable physical alphabet whose tokens preserve rendering and spacetime structure.",
                "The target is a stable physical alphabet—not a codec bit rate.",
                "comparison",
                {"left": "raw fitted rows", "right": "canonical Jewel phrases", "left_value": "non-unique", "right_value": "shared tokens"},
            ),
            Shot(
                "Rotation has gauge ambiguity",
                "A unit quaternion q and its negation represent exactly the same rotation. Even after choosing quaternion sign, covariance eigenvectors may permute when their eigenvalues are reordered, and each eigenvector admits a sign flip. Feeding scale plus quaternion to a prior makes identical ellipsoids appear far apart. This is pure representational noise: it changes parameter coordinates without changing a single rendered pixel.",
                "q and −q render identically; covariance axes also permute and flip.",
                "equation",
                {"equation": "R(q) = R(-q),   Sigma = R diag(s^2) R^T", "diagram": "gauge"},
            ),
            Shot(
                "Log-covariance removes the gauge",
                "The canonical representation stores the matrix logarithm of the symmetric positive-definite covariance. A symmetric three-by-three matrix needs six unique numbers. The logarithm maps multiplicative scale differences into a well-behaved Euclidean coordinate chart. Decoding exponentiates, eigendecomposes, and converts one valid rotation back to the renderer. Any sign or axis choice made during that last factorization renders the same covariance.",
                "log Σ is unique and symmetric; six numbers replace scale-plus-quaternion gauge.",
                "feature-vector",
                {"segments": [("μ", 3, 0), ("log Σ", 6, 1), ("RGB", 3, 2), ("JRGB", 9, 3), ("a", 1, 4)]},
            ),
            Shot(
                "One joint token was too rigid",
                "The first vocabulary treated an eight-Jewel constellation as one joint one-hundred-and-seventy-six-dimensional prototype. That couples layout, covariance, surface color, and color gradient into one indivisible choice. Gate zero-a falsified that design. The successful language assigns independent codebooks to physical roles, then composes their aligned prototypes back into the same constellation. Factorization lets reusable geometry combine with reusable appearance.",
                "Reject one 176-D joint token; compose independent physical roles.",
                "factorization",
                {"factors": [("layout", "K=1024", 0), ("covariance", "K=1024", 1), ("surface", "K=1024", 2), ("gradient", "K=1024", 3)]},
            ),
            Shot(
                "Combinatorial capacity",
                "Four one-thousand-and-twenty-four-way decisions expose up to one-thousand-and-twenty-four to the fourth role combinations without fitting that impossibly large joint table. For individual generated Jewels, the centroid remains continuous and the three nonconstant roles—covariance, surface, and gradient—are active tokens. Layout becomes the spoken centroid itself. The codebooks share normalization and preserve the canonical twenty-two-feature contract.",
                "4 codebooks × K=1024 expose compositional capacity without a K^4 table.",
                "tokens",
                {"tokens": ["covariance", "surface", "gradient"], "values": ["cov=418", "rgb=77", "grad=903"]},
            ),
            Shot(
                "Addresses are internal; centers are continuous",
                "Cells are useful for learning local histograms and addressed phrases, but an address is not an emitted coordinate. The generator outputs centroids in continuous normalized space. Internal cell indices answer questions like which local token distribution applies here. They do not snap a Jewel to the cell center. This separation directly addresses the grid-quantization artifact: routing may be discrete while geometry remains irregular.",
                "discrete routing cell ≠ emitted centroid; μ stays continuous.",
                "spacetime",
                {"mode": "continuous", "foreground": 0, "background": 0},
            ),
            Shot(
                "Gate zero-f: the physical alphabet survives",
                "At the selected individual-Jewel language gate, decoded irregular fields retain twenty-two-point-eight-six-five-seven decibels against the continuous source on the frozen random-volume audit, preserve mixed spacetime tilt at one-point-zero-four-one-five times the source, and show zero center locking. These numbers do not prove promptability. They prove that the physical token vocabulary is not the bottleneck preventing a higher-level speaker from producing renderable continuous fields.",
                "22.8657 dB · 1.0415× tilt retention · 0% grid locking",
                "metrics",
                {"bars": [("PSNR", 22.8657, 25.0, 0), ("tilt retention", 1.0415, 1.2, 1), ("grid lock", 0.0, 1.0, 4)]},
            ),
        ),
    ),
    Episode(
        4,
        "coherence-trajectories",
        "Why coherence needs persistent ownership",
        "The failure ladder that led to trajectory programs",
        (
            "sol/block_token_jewel_speaker.py",
            "sol/coherent_source_realizer.py",
            "sol/semantic_trajectory_realizer.py",
            "sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md",
        ),
        (
            Shot(
                "Independent Jewels make texture",
                "The naïve generator predicts each Jewel mark from global text and local coordinates. It can learn marginal colors, covariance frequencies, and density. What it does not own is a persistent object. Independent local choices are statistically plausible but mutually inconsistent: one region votes for fur, another for metal, and successive time slabs disagree about identity. The result is structured texture rather than a recognizable actor performing an action.",
                "Correct local marginals do not imply a coherent global subject.",
                "comparison",
                {"left": "independent marks", "right": "persistent owner", "left_value": "texture", "right_value": "subject"},
            ),
            Shot(
                "Local block phrases were better—but insufficient",
                "Addressing a sixteen-by-sixteen-by-eight spacetime grid and predicting one of one-thousand-and-twenty-four local block phrases lowers direct token negative log likelihood to five-point-one-six-eight, fifteen-point-six percent better than the global posterior. Yet qualitative renders remain texture. That experiment is vital because it separates two hypotheses. Local statistical modeling works. Recognizable composition still fails because no variable binds those local phrases into one object trajectory.",
                "Addressed phrases: NLL 5.168, 15.60% better—still no subject.",
                "metrics",
                {"bars": [("global NLL", 6.1234, 7.0, 4), ("local NLL", 5.1680, 7.0, 1), ("recognizable", 0.0, 3.0, 5)]},
            ),
            Shot(
                "One coherent owner restores the subject",
                "The next oracle chooses one complete training-owned program for the entire spacetime window. Nothing about the physical vocabulary changes. The same continuous centroids and one-thousand-and-twenty-four-way active tokens suddenly render recognizable ballerina, dog, and welder examples. This is retrieval-only and therefore not the final generator. But causally, it identifies persistent cross-block ownership as the missing variable.",
                "Same physical Jewels + one persistent owner → all three subjects return.",
                "evidence",
                {"asset": "coherent", "label": "coherent-source source-disjoint audit"},
            ),
            Shot(
                "Two donors rule out whole-video retrieval",
                "A complete-source owner could be dismissed as nearest-neighbor playback. So the next experiment composes two distinct training programs. A moving foreground tube selects material from one donor; everything outside is supplied by another. Wrong-object controls deliberately take the foreground donor from a different semantic scene while preserving the path. The visible subject swaps. The field is therefore a new composition, and the tube has causal semantic ownership.",
                "Distinct foreground and background donors form a new field.",
                "spacetime",
                {"mode": "donors", "foreground": 36000, "background": 36000},
            ),
            Shot(
                "The semantic path comes from training-only saliency",
                "For each addressed block, the realizer compares the scene's mean normalized descriptor with the mean of all other scenes and squares the difference. A centered spatial prior suppresses unstable borders. Summing saliency over u and v yields one center per time slab, then a one-two-one temporal filter smooths the path. The path therefore follows where a class differs from alternatives, not where two arbitrary source fields happen to disagree.",
                "path(t) = center of scene-vs-other descriptor saliency, smoothed [1,2,1]",
                "equation",
                {"equation": "s_k = mean_d (D_scene[k,d] - D_other[k,d])^2;  p(t) = weighted_center(s)", "diagram": "path"},
            ),
            Shot(
                "Rank balance fixes density mismatch",
                "A single geometric radius can select unequal totals because different valid fields have different local densities. Instead, Gate two-b-zero sorts foreground Jewels by increasing squared tube distance and takes the closest half-budget. It sorts background Jewels in the opposite direction and takes the farthest half-budget. Top-k selection makes the emitted count exact, forces fifty-fifty ownership, and requires no adjustment duplicates or target-derived threshold.",
                "top-k nearest foreground + top-k farthest background; no density assumption.",
                "equation",
                {"equation": "F = arg top_36000(-d_tube^2);  B = arg top_36000(+d_tube^2)", "diagram": "rank"},
            ),
            Shot(
                "Causal controls, not one pretty sample",
                "Every intended prompt is rendered under three seeds. Under the same declared seed, a cyclic-shuffled prompt compiles the next semantic scene, while a null prompt derives scene from seed alone. Correct programs must beat those generated controls, not merely score against alternative text on an unchanged render. This distinction matters: it tests whether changing text causes the generated field to change in the intended direction.",
                "Correct, cyclic-shuffled, and null prompts generate matched causal controls.",
                "evidence",
                {"asset": "proof-sheet", "label": "matched correct-versus-shuffled middle frames"},
            ),
        ),
    ),
    Episode(
        5,
        "evidence-gates",
        "What the experiments actually prove",
        "Exact gate, learned speaker, plateau, and honest failure scope",
        (
            "sol/audit_prompt_trajectory_speaker.py",
            "sol/train_learned_trajectory_speaker.py",
            "sol/audit_learned_trajectory_speaker.py",
            "sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md",
        ),
        (
            Shot(
                "Freeze the gate before looking",
                "The exact audit freezes prompts, seeds, field size, renderer, frame indices, semantic evaluator, and pass thresholds before execution. It renders frames zero, twenty-four, and forty-eight at one-forty-four by two-sixteen. OpenCLIP ViT-B thirty-two embeds each frame, normalized frame embeddings are mean-pooled, and the resulting video vector is compared against the three exact prompts. The preregistration prevents a visually appealing seed from redefining success.",
                "3 prompts × 3 seeds × 3 causal conditions; frozen before execution.",
                "pipeline",
                {"nodes": ["3 frames", "OpenCLIP", "mean unit vector", "prompt scores"]},
            ),
            Shot(
                "Retrieval and generation margins test different things",
                "Top-one retrieval asks whether a correct generated field is closest to its intended prompt among three choices. The shuffled generation margin instead holds the intended text fixed and compares a correctly conditioned field against a field actually generated from wrong text. Null margin does the same with no semantic prompt. Pairwise win counts retain seed-level information that a mean could hide. All are needed because any one semantic score can be fooled by class imbalance or evaluator bias.",
                "retrieval: rank text for one field; margin: change the generated field itself.",
                "comparison",
                {"left": "same render / texts", "right": "same text / generations", "left_value": "retrieval", "right_value": "causality"},
            ),
            Shot(
                "Exact Gate two-b-zero passes",
                "Across nine exact prompt programs, intended retrieval succeeds eight times. Every prompt class has majority retrieval. Mean correct-minus-shuffled generation similarity is plus zero-point-zero-five-six-four-eight, above the frozen plus zero-point-zero-two threshold. Correct-minus-null is plus zero-point-zero-three-three-four-eight, above plus zero-point-zero-one. Correct wins all nine shuffled and all nine null pairs. Counts, donor distinction, finite rendering, and continuous-center checks also pass.",
                "8/9 retrieval · +0.05648 shuffled · +0.03348 null · 9/9 wins",
                "metrics",
                {"bars": [("retrieval", 8, 9, 0), ("shuffled wins", 9, 9, 1), ("null wins", 9, 9, 2), ("grid lock", 0, 1, 4)]},
            ),
            Shot(
                "The learned speaker is deliberately small",
                "Gate two-b-one replaces exact lookup with a five-hundred-forty-one-thousand-two-hundred-twenty-three-parameter autoregressive network. Frozen text embeddings enter a two-layer projection. A scene head predicts one of three semantic tokens. Scene embedding conditions the foreground head. Text, scene, and sampled foreground condition the background head. Source logits are not masked by scene; only exact foreground-background repetition is forbidden. Scene-consistent donors must therefore be learned rather than imposed.",
                "text → scene → foreground → background; only donor repetition is masked.",
                "pipeline",
                {"nodes": ["text embedding", "scene logits", "foreground logits", "background logits"]},
            ),
            Shot(
                "Held-out wording and held-out donor pairs",
                "Training uses two authored paraphrases plus the exact prompt for each class and enumerates ordered donor pairs, while reserving cyclic pairs for evaluation. The held-out set changes both wording and foreground-background combination. Ten percent empty-text dropout teaches a null condition. Correct, cyclic-shuffled, and empty embeddings score the identical target programs, which makes token negative log likelihood a clean conditional test before any renderer or CLIP model enters the loop.",
                "Evaluation changes both paraphrase and donor combination.",
                "tokens",
                {"tokens": ["train wording", "held-out wording", "held-out pair"], "values": ["3/class", "1/class", "18 programs"]},
            ),
            Shot(
                "Longer training did not solve the learned gate",
                "The best correct-condition negative log likelihood occurs at step one hundred. Evaluation continues every one hundred updates. Ten consecutive evaluations through step eleven hundred fail to improve it, satisfying the frozen plateau rule. Correct held-out NLL is two-point-two-zero-zero-five, compared with four-point-three-eight-seven under cyclic-shuffled wording and two-point-five-three-nine-one for null. Scene accuracy is one hundred percent for correct unseen paraphrases. This is evidence of early overfit, not evidence that the run was simply too short.",
                "best step 100; ten stale evaluations; stopped at 1,100 by the frozen rule.",
                "curve",
                {"series": [("correct", [2.20, 2.32, 2.47, 2.58, 2.66, 2.73], 1), ("shuffled", [4.39, 4.55, 4.72, 4.83, 4.92, 5.02], 4), ("null", [2.54, 2.61, 2.68, 2.73, 2.78, 2.82], 0)]},
            ),
            Shot(
                "The learned result is a scoped near-pass",
                "All nine correct-text samples predict the right scene and emit both unmasked donors from that scene. Rendered correct-minus-shuffled and correct-minus-null margins pass, and correct beats shuffled in seven of nine. But raw three-way OpenCLIP top-one is only four of nine, below the required six, with majority retrieval in only one class. We retain that failure. The learned network understands the tiny program syntax and causal scene conditioning; it does not yet establish robust open-vocabulary semantic generation.",
                "Token syntax and causal margins pass; strict learned retrieval remains 4/9 and fails.",
                "evidence",
                {"asset": "evidence", "label": "exact pass and learned near-pass, with failed gate retained"},
            ),
        ),
    ),
    Episode(
        6,
        "scaling-to-t2v",
        "From bounded proof to full text-to-video",
        "The concrete data-and-compute program",
        (
            "sol/results/jewel_casting_language_v0/TRAJECTORY_SPEAKER_REPORT.md",
            "sol/results/jewel_casting_language_v0/PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md",
            ".beads/issues.jsonl",
        ),
        (
            Shot(
                "What the proof buys us",
                "The project now demonstrates every boundary of a hierarchical native speaker except one. Text and seed can choose a finite program. Persistent tokens can own a coherent trajectory. A program can expand to exact-count continuous Jewels. A factorized physical vocabulary can render an irregular spacetime field. Causal text controls beat shuffled and null generations. The missing step is replacing source-backed macro tokens with reusable learned object, motion, and background prototypes.",
                "The mechanism is connected end-to-end; macro-token reuse is the missing scale step.",
                "pipeline",
                {"nodes": ["text", "persistent program", "physical tokens", "continuous field"]},
            ),
            Shot(
                "The source-token bottleneck",
                "Today, foreground token seven ultimately expands to a constellation derived from one training field. Composing it with a distinct background proves that the system is not retrieving one complete video, but it still limits novelty. Scaling eighteen source identifiers to eighteen thousand would memorize more examples, not discover a language. The next vocabulary must pool recurring connected spacetime tubes across many videos and assign prototypes independent of source identity.",
                "Do not scale source IDs; learn reusable connected trajectory prototypes.",
                "comparison",
                {"left": "source token 7", "right": "learned prototype 7", "left_value": "one field", "right_value": "many fields"},
            ),
            Shot(
                "Learn sixty-four to two-hundred-fifty-six prototypes",
                "The next experiment fits at least one hundred prompted fields spanning ten to twenty compositional prompts. It learns sixty-four to two-hundred-fifty-six foreground and background trajectory prototypes. Each prototype owns a connected tube, but decodes through the existing one-thousand-and-twenty-four-way covariance, surface, and gradient vocabularies. A no-copy audit rejects any generated macro token that references a source identifier or inherits more than a preregistered fraction from one field.",
                "100+ fields · 10–20 prompts · 64–256 reusable trajectory prototypes",
                "future",
                {"stages": ["100+ fitted fields", "connected tube mining", "64–256 prototypes", "no-source-ID audit"]},
            ),
            Shot(
                "A typed program transformer",
                "The full speaker should not autoregress seventy-two thousand independent Jewels. It should emit a typed hierarchy: scene, style, and camera; persistent object and background identities; trajectory anchors and derivatives; addressed local constellation phrases; then continuous centroids and physical tokens. A small transformer can model the discrete program. Specialized differentiable realizers can enforce geometry, exact counts, and support-safe rendering. This division spends sequence-model capacity on semantic dependencies instead of raster-like repetition.",
                "scene → objects → path derivatives → local phrases → continuous Jewels",
                "pipeline",
                {"nodes": ["scene/style/camera", "object tracks", "path derivatives", "local phrases", "Jewels"]},
            ),
            Shot(
                "Derivatives make motion compositional",
                "A class-level path is too rigid for a complete model. Instead the speaker should emit trajectory anchors together with velocity, and eventually acceleration or spline control points. Position in the next window is predicted from carried state rather than reset from a template. Derivatives compactly express direction and continuity, while local Jewel phrases decorate the moving tube with appearance and geometry. This is why the derivative method outperformed raw local coordinates in the earlier experiments: it assigns the model the temporally stable quantity.",
                "carry state: position, velocity, identity, style—not a fresh independent window.",
                "equation",
                {"equation": "p[k+1] = p[k] + dt*v[k] + 0.5*dt^2*a[k];  v[k+1] = v[k] + dt*a[k]", "diagram": "derivative"},
            ),
            Shot(
                "The next gate must test composition and carry",
                "Training and evaluation prompts must be split by object-action combination, not random video. Correct text must beat cyclic object and action swaps on held-out compositions. Every prompt needs at least two recognizable but structurally distinct samples. A second window must continue identity and velocity from generated carry alone, with no target field or teacher scaffold. These gates distinguish real language scaling from memorization, evaluator bias, and single-window tricks.",
                "Held-out compositions, diverse samples, no source copying, generated second-window carry.",
                "exclusion",
                {"allowed": ["prompt", "seed", "generated carry"], "forbidden": ["source ID", "target field", "teacher scaffold", "seen object-action pair"]},
            ),
            Shot(
                "The honest compute pitch",
                "More compute is useful only after the representation assigns it the right job. Data buys repeated examples of objects, actions, cameras, and trajectories. Prototype learning converts those repetitions into reusable macro tokens. Program-model compute learns prompt-to-token composition and long-range dependencies. Rendering compute improves local Jewel realization and support indexing. The pitch is therefore not that a larger model will rescue independent splats. It is that every interface is now demonstrated, and scale targets the one unproven interface: reusable vocabulary breadth and cross-window program modeling.",
                "Compute → paired fields → reusable prototypes → broader programs → longer coherent video",
                "future",
                {"stages": ["paired data", "prototype vocabulary", "program transformer", "generated carry", "open-vocabulary T2V"]},
            ),
        ),
    ),
)


def episode_by_number(number: int) -> Episode:
    try:
        return next(episode for episode in EPISODES if episode.number == number)
    except StopIteration as error:
        raise ValueError(f"unknown episode {number}") from error
