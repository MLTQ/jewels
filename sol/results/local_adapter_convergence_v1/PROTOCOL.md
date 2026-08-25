# Frozen appearance-adapter convergence protocol

## Question

Did the 400-update appearance-adapter screen fail because the representation was insufficient, or
because each clip received too little optimization exposure?

## Fixed factors

- Frozen source: replicated seed-0 frozen residual checkpoint.
- Training corpus: 120 clips.
- Selection audit: five cooking clips spanning anime, cartoon, clay, photoreal, and render3d.
- Exact evaluation: seven fixed frames per clip through the support-correct tiled renderer.
- Objective: render reconstruction plus train-only LPIPS weight `0.05`.
- Ownership: geometry and every non-adapter parameter must remain bitwise source-equal.
- Quality floor: exact macro PSNR at least `20 dB`.
- Positive appearance threshold: exact macro LPIPS below `0.70`.

## Runs

- Raw local radius-2/time-1 adapter, seed 0: 12k updates plus a 4k low-rate continuation.
- Forced derivative-x32 radius-2/time-1 adapter, seed 0: 12k updates.
- Forced derivative-x32 radius-2/time-1 adapter, seed 1: independent 12k replication.

Checkpoints were retained at registered milestones. The dense derivative audit uses source, 400,
800, 1.6k, 3.2k, 4k, 8k, and 12k. The replication audit uses source plus 4k/8k/12k from both seeds.

## Stopping rule

Continue while exact held-out LPIPS still improves materially. Stop an arm after two consecutive
evaluation intervals show less than `0.5%` relative validation improvement and exact LPIPS changes
by less than `0.001`, or stop immediately for exact PSNR below `20 dB`, visible collapse, or loss of
frozen-parameter ownership.

The raw arm received an additional 4k continuation because its 12k endpoint was initially
ambiguous. Its 12k→16k LPIPS improvement was `0.00021`. Derivative seed 0 improved `0.00072` from
8k→12k; seed 1 improved `0.00100` after rounding. Together with flat sampled validation and stable
qualitative images, this is sufficient evidence of an 8k–12k plateau for the current schedule.

## Interpretation constraint

The exact selection clips were already observed during earlier architecture work, so these results
may select a mechanism but cannot estimate final generalization. A third seed and fresh unseen
clips/prompts are required before making a broad model-quality claim. Corpus size is unchanged, so
this protocol demonstrates compute scaling, not data scaling.
