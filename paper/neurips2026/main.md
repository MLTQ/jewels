# NeurIPS concept-and-feasibility manuscript

## Intent

The main LaTeX source is the concise, reviewer-facing paper for the project's strongest current
claim. It is separate from the exhaustive experiment report in the parent paper directory.

The argument is intentionally asymmetric:

1. It trusts replicated positive evidence and states the exact population tested.
2. It reports failed gates as limits on a configuration, not universal impossibility claims.
3. It exposes the source-backed macro vocabulary as the central unresolved dependency.
4. It rejects codec or compression framing; Jewels are an executable native output space.

## Scientific claim

The demonstrated claim is that exact prompt text plus an integer seed can select a hierarchical
source-backed Jewel program and render a prompt-selective video without a target video, target
field, target program, or held-out latent at inference. It is not an open-vocabulary text-to-video
claim.

The learned speaker establishes learnability of the coarse program syntax on unseen paraphrases,
but its 4/9 strict rendered OpenCLIP top-1 result fails the declared 6/9 gate. This negative result
must remain visible.

## Evidence contracts

- Temporal tilt: 3 sources by 3 seeds, full covariance versus projected zero space--time terms.
- Encoder scaling: nested 12/60/120 train sets, three seeds, fixed 60-video validation inventory.
- Physical vocabulary: three independent 1024-entry books, exact continuous centroids.
- Exact prompt speaker: three classes by three seeds with correct, cyclic-shuffled, and null text.
- Learned speaker: unseen fourth paraphrase for each class, frozen best-NLL checkpoint.

Numerical claims must remain traceable to the paths listed in the reproducibility appendix. If
experiments are rerun, update the manuscript, evidence table, checklist justifications, and figure
captions together.

## Layout and submission contract

- Use the unmodified official NeurIPS 2026 style in anonymous Main Track mode.
- Main content must occupy no more than nine pages; references, appendices, and checklist follow.
- The architecture figure must distinguish demonstrated source-backed macros from the proposed
  source-independent vocabulary.
- Render the PDF to page images and inspect all pages before delivery.

## Future revision gates

Before describing the system as open-vocabulary text-to-video, require all of the following:

- macro and trajectory prototypes are learned across sources rather than keyed to fitted fields;
- entire motions, appearances, and their compositions are held out;
- correct, shuffled, and null prompt margins replicate across substantially more classes and seeds;
- video quality is evaluated beyond frozen CLIP screening; and
- direct continuous and dense-latent baselines are compared at matched compute and quality.
