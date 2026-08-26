# Jewel Field: Technical Explainer Series

Six original, narrated mathematical-animation chapters explain the current Jewel text-to-video
proof, the exact Gaussian field renderer, the native token language, the coherence experiments,
the frozen evidence gates, and the path to a scalable promptable model.

The visual language uses a dark vector canvas, animated diagrams, equations, evidence plots, and
retained generated clips. It is inspired by the clarity of modern mathematical animation while
remaining an original design rather than copying another channel's branding, assets, or voice.

## Watch in order

1. [From words to a spacetime program](episode_01_prompt-to-program.mp4) — 3:46
2. [A Jewel is a Gaussian in spacetime](episode_02_jewel-geometry.mp4) — 3:21
3. [Turning Jewels into a native language](episode_03_native-vocabulary.mp4) — 3:20
4. [Why coherence needs persistent ownership](episode_04_coherence-trajectories.mp4) — 3:18
5. [What the experiments actually prove](episode_05_evidence-gates.mp4) — 3:45
6. [From bounded proof to full text-to-video](episode_06_scaling-to-t2v.mp4) — 3:55

Total runtime: approximately 21 minutes 25 seconds.

Each MP4 is 1280×720 H.264 with mono AAC narration and embedded English subtitles. Matching
sidecar `.srt` files, verbatim narration scripts, and posters are included for editing and review.
The complete media/provenance contract is recorded in [inventory.json](inventory.json).

## What the series claims

The current evidence proves a bounded causal path from registered text prompts to native Jewel
programs, continuous irregular fields, and rendered video. It does **not** claim broad open-vocabulary
text-to-video. The final two chapters deliberately separate exact-program success from the learned
speaker's partial generalization, then name the remaining macro-token data bottleneck and frozen
gates required to support a larger compute pitch.

## Reproduce

From the repository root, run:

```bash
/Users/max/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m sol.render_jewel_explainer \
  --project-root "$PWD" \
  --output sol/results/jewel_explainer_series_v1 \
  --fps 12
```

The renderer uses local macOS speech synthesis and `ffmpeg`; it does not use a network voice service
or either GPU. The frozen episode specifications and citations live in
`sol/jewel_explainer_episodes.py`.

## Editorial note

The included Daniel system voice is a reproducible review track. Before public release, a human
narration pass and a final terminology review would improve presentation without changing the
technical animation or evidence contract.
