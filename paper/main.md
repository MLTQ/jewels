# `main.tex`

## Purpose

Maintains the project research paper and the exact distinction between completed evidence,
negative results, and future claims. The compiled PDF is a technical progress report rather than a
submission-ready benchmark paper.

## Components

### Literature and claim scope

- **Does**: Positions Jewels against Gaussian video reconstruction, video codecs, dynamic 3D
  Gaussian scenes, and generative 3D Gaussian models.
- **Interacts with**: `references.bib` and the repository's August 2026 literature audit.
- **Rationale**: Video reconstruction with Gaussian splats predates this project; only the narrower
  persistent, generative, locally repairable `(u,v,t)` field direction is provisionally open.

### Method

- **Does**: Defines the additive P1 jewel renderer, gauge-free log-covariance features, dense
  tokenizer hierarchy, rectified-flow prior, stable-ID continuation, scaffold-conditioned
  occupancy/positive-count decoding, and exact edit clamping.
- **Interacts with**: Production contracts documented in `stprim/` and experimental contracts in
  `sol/`.

### Experiments and discoveries

- **Does**: Consolidates reconstruction, density, tokenizer, generation, continuation, editing, and
  prompt-preflight results, including failed controls. The current report includes the exact
  topology/mark-group washout decomposition, oracle-topology stochastic flow, and privileged
  low-resolution video-guide experiment, the 16-video LTX scaffold gate, all four density-matched
  jewel fits, the negative token-only and residual-hybrid controls, the non-dominating raster
  render-loss sweep, the passing UCF-train/LTX-validation realizer transfer gate, and the first
  learned-topology/frozen-realizer continuation with exact carry and density controls.
- **Interacts with**: Figures and metric reports under `assets/` and `sol/results/`.

### Semantic scaffold architecture

- **Does**: Records why occupancy/count prediction is no longer the immediate washout fix and
  specifies a pretrained-video-scaffold to stochastic-jewelizer path.
- **Interacts with**: `audit_prompted_washout.py`, `birth_mark_flow.py`, `video_guide.py`, and the
  prompt-smoke result artifacts.
- **Rationale**: Exact topology adds only 0.172 dB, stochasticity restores detail without coherence,
  a privileged `24x40` guide restores macro-layout and 90.5% of target edge energy, and the first
  local-token branches show that within-cell detail cannot replace or simply stack onto
  cross-cell/global guide context. The lower render-loss weight raises SSIM/edge detail but
  destabilizes Guitar, so v1 remains selected. Without retraining, v1 then reaches 15.342 dB /
  0.6942 SSIM on four unseen LTX scaffolds versus 13.777 dB / 0.4160 deterministic continuation,
  passing the privileged transfer gate. The new topology head causally beats shuffled/null count
  controls and, after synthesizing every predicted rank, reaches 15.464 dB / 0.6967 SSIM and 5,822
  effective contributors/frame within 0.028 dB of oracle topology. Initial state generation and
  multi-stride generated carry remain open.

### Learned topology continuation

- **Does**: Records the 0.88M scaffold/carry topology head, UCF-only threshold calibration,
  correct/shuffled/null/train-mean controls, the invalidity of exact fitted-rank retention as a
  generative density metric, and the decisive frozen-realizer rendering of every predicted rank.
- **Rationale**: Learned topology predicts 99.68% of the held-out continuation birth budget,
  averages 5,822 effective contributors/frame, copies carry at zero error, and reaches 15.464 dB /
  0.6967 SSIM versus 15.492 dB / 0.6997 under oracle topology. Shared learned/oracle noise moves the
  immediate quality bottleneck to mark realization.

### Limitations and path forward

- **Does**: Prevents unfinished prompt generation or repair quality from being described as solved
  and specifies the gates required for a future novelty claim. It now distinguishes the passing
  one-stride learned-topology result from still-open initial generation, generated multi-stride
  carry, weak rendered topology selectivity, and foreground/motion mark noise.

### Representation rationale and multimodal event language

- **Does**: Separates the explicit jewel field from the diffusion, flow, or autoregressive
  objective used to generate it; states the interaction advantages and their failure gates.
- **Interacts with**: The stable-ID streaming contract, masked repair, the `16^3` hierarchy, and
  primary multimodal work cited in `references.bib`.
- **Rationale**: The defensible opportunity is persistent executable visual state, not a claim that
  LLM-native image or video generation is itself new.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Compiled report | Figure paths remain valid relative to `paper/` | Moving result artifacts |
| Novelty claims | Broad first-reconstruction claim remains explicitly rejected | New prior art or completed comparison |
| Multimodal framing | Existing token-native image/video generation remains credited | New multimodal prior art or event schema |
| Result tables | Values match tracked JSON/README evidence | Reruns with superseding protocols |
| PDF QA | ASCII source text and compilable bibliography | Unsupported TeX packages or Unicode |

## Notes

- Update the report date and literature cutoff whenever results or claims change.
- Author names are intentionally represented as “The Jewels Project” until the human authorship
  and affiliation block is finalized.
- The final PDF is built as `output/pdf/jewels_progress_report.pdf` and visually inspected page by
  page before delivery.
- The compact related-work table uses footnote-sized type so long method distinctions remain inside
  the text block without abbreviating the substantive comparison.
- Figures and tables use fixed in-order placement because deferred research figures can otherwise
  migrate behind the bibliography and detach evidence from the claims it supports.
- Quantitative scaling and qualitative samples are separate figures so the evidence remains
  legible while filling pages naturally.
- The diffusion comparison is representation-level: jewels may still be generated by diffusion or
  flow matching. The event-language proposal remains conditional on compression, held-out prompt
  control, persistent rollout, and cheaper local edits.
- The prompt-generated video guide is part of the selected hybrid architecture, but one learned
  continuation does not license a direct prompt-only jewel model or an end-to-end persistent run.
- LTX scaffold generation, optimized jewel reconstruction, topology prediction, and learned mark
  realization remain separate metrics. Cross-domain realization and one learned-topology stride
  now pass, but correct/shuffled text remains tied under the guide and initial/multi-stride jewel
  generation remains open.
