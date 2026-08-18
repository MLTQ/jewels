# Stage 1 v0: text-conditioned generation in the encoder latent — FAILS G3

First attempt at pulling generation into our own stack: an 8.45M rectified flow over the frozen
encoder's cell-token latent (2,048 tokens x 364 channels), conditioned on prompt **token
sequences** via cross-attention with classifier-free dropout, trained on 180 windows spanning
12 action classes x 5 visual styles. Decoding uses the frozen amortized encoder, so joint
composition is the decoder's job.

## Verdict: text conditioning does not control generation here

| Test | correct | shuffled | margin | reading |
|---|---:|---:|---:|---|
| Held-out latent velocity MSE | 0.45181 | 0.45202 | 0.0002 | nil (null arm is 0.00007 *better*) |
| Rendered, photoreal-only (12) | 10.695 dB | 10.458 dB | +0.24 dB, 10/12 | weak (p~0.02) |
| **Rendered, style-stratified (15)** | **10.132 dB** | **10.090 dB** | **+0.04 dB, 9/15** | **chance (p~0.30)** |

The photoreal-only margin did not survive evaluating across all five styles; it reflected
homogeneous content, not conditioning. Absolute quality is also unusable — 10.1 dB / 0.030 SSIM
against the encoder's own 23.41 dB / 0.9436 on the same corpus, and below even the retired
mark-space stack (14.5 dB / 0.60).

## Why: the latent is ~90% text-unpredictable detail

Permutation-tested variance explained in the cached latents (chance level from 12 label
shuffles, essential because 96 prompt groups over 240 windows explains ~39% by construction):

| grouping | observed | chance | ratio |
|---|---:|---:|---:|
| style (5) | 0.201 | 0.017 | **11.8x** |
| class (12) | 0.094 | 0.045 | 2.1x |
| prompt (96) | 0.424 | 0.394 | 1.1x |

Style is strongly encoded, action weakly, prompt-level detail not at all. With ~90% of latent
variance instance-specific, an MSE flow converges toward the conditional mean and renders as
washout — the same conditional-averaging failure this project documented at the mark level,
reproduced one level up. That reproduction is the useful part: the problem is not *where*
generation happens (marks vs latent) but that the model is asked to hallucinate high-entropy
detail its conditioning cannot determine.

## Consequence: factorize instead of scaling

The latent splits almost evenly (cells 49% / seed 51% of variance) and `seed` is literally a
coarse 48x48x24 RGB volume on the slot lattice. The strong text signal (style) is exactly what
such a coarse volume carries. The well-posed decomposition is therefore

    text -> seed (coarse video)  ->  cells (detail)  ->  jewels

where the text model only produces what text determines, and detail comes from learned
refinement rather than sampling noise. This is the teacher-scaffold architecture with our own
small scaffold generator, which Stage 2's distillation step required regardless.

Do **not** respond to this result by scaling the current design: more data or capacity cannot
make text predict variance that text does not carry.

Artifacts: `train_summary.json`, `render_gate_report.json` (photoreal-only),
`render_gate_styles_report.json` (authoritative, style-stratified).
