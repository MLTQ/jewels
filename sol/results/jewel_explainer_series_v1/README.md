# Jewel Field: Plain-Language Explainer Series

Six narrated visual chapters explain the current Jewel text-to-video proof, what a spacetime
Gaussian Jewel is, why the model needs a native physical vocabulary, how persistent trajectories
restore coherence, what the experiments genuinely prove, and what data-and-compute step comes next.

The scripts use one idea per shot, define technical terms on first use, and prefer ordinary words
afterward. Analogies carry the geometry: a video is a stack of transparent frame sheets; a Jewel is
a soft blob passing through that stack; a prompt becomes a short recipe; and a moving subject owns
a tube through time. Exact equations and source paths remain available for audit without dominating
the spoken explanation.

## Watch in order

1. [How a prompt becomes a video](episode_01_prompt-to-program.mp4) — 2:36
2. [What is a spacetime Gaussian Jewel?](episode_02_jewel-geometry.mp4) — 2:39
3. [Giving the model a Jewel vocabulary](episode_03_native-vocabulary.mp4) — 2:41
4. [Why motion needs a persistent owner](episode_04_coherence-trajectories.mp4) — 2:30
5. [How we decide whether it really works](episode_05_evidence-gates.mp4) — 2:37
6. [What would turn this into text-to-video?](episode_06_scaling-to-t2v.mp4) — 3:01

Total runtime: 16 minutes 4 seconds.

Each MP4 is 1280×720 H.264 with mono AAC narration and embedded English subtitles. Matching
sidecar `.srt` files, verbatim narration scripts, and posters are included for editing and review.
The [six-episode contact sheet](series_contact_sheet.png),
[42-shot QA sheet](all_shots_contact_sheet.png), and complete
[media/provenance inventory](inventory.json) support visual and technical review.

## What JRGB meant

Earlier screens used `JRGB` for the three-by-three local color Jacobian: how red, green, and blue
change when moving left-right, up-down, or forward in time near one Jewel. The revised public series
removes that abbreviation. Episode 2 introduces the full term once and then calls it the
“color-change table.” Covariance remains separate: it controls the Jewel's shape, persistence, and
space-time tilt, while the color Jacobian controls local appearance change.

## What the series claims

The evidence proves a bounded causal path from three registered text prompts to native Jewel plans,
continuous irregular fields, and rendered video. It does **not** prove broad, open-vocabulary
text-to-video. The series keeps the learned model's strict 4-of-9 failure visible and explains why
the next decisive experiment is a larger, source-independent vocabulary learned from repeated
concepts across at least one hundred varied Jewel fields.

## Reproduce

The default production path uses the Pharaoh Qwen3-TTS Base model at the configured LAN endpoint and
the committed original officer-style reference audio:

```bash
/Users/max/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m sol.render_jewel_explainer \
  --project-root "$PWD" \
  --output sol/results/jewel_explainer_series_v1 \
  --fps 12 \
  --tts-backend qwen-clone \
  --qwen-url http://192.168.0.202:18001
```

Every shot records its deterministic seed, timing bound, audio-token ceiling, accepted attempt,
reference hash, and synthesis settings in `inventory.json`. The renderer rejects abnormally short,
long, or token-cap-hitting takes. A macOS `say` backend remains available only as a local fallback.

## Narration identity

The reference is an original warm American first-officer archetype created for this project. It is
not an imitation of a real actor or a fictional character performance. All 42 production takes were
accepted on their first attempt, normalized to approximately -18 LUFS, and verified to contain no
silence lasting two seconds or more.
