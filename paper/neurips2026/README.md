# NeurIPS 2026 concept-and-feasibility paper

## Purpose

This directory contains a submission-style paper that distills the project evidence into one
bounded claim: a prompt and seed can select a hierarchical native Jewel program and render a
recognizable prompt-selective video without a target video or target Jewel field at inference.
It does **not** claim an open-vocabulary or production text-to-video model.

The older `paper/main.tex` remains the exhaustive technical progress report. This manuscript is a
shorter argument organized for the NeurIPS 2026 Main Track contribution type “Concept &
Feasibility.”

## Files

- `main.tex`: anonymous submission manuscript.
- `checklist.tex`: mandatory NeurIPS checklist with current answers.
- `neurips_2026.sty`: unmodified official 2026 style file.
- `figures/architecture_overview.tex`: demonstrated architecture and next gate.
- `figures/jewel_geometry.tex`: one Jewel in `(u, v, t)` and its parameters.
- Companion Markdown files document the intent and contracts of each maintained source file.

## Build

From the repository root:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=jewels_neurips2026_concept_feasibility \
  paper/neurips2026/main.tex
cp paper/neurips2026/jewels_neurips2026_concept_feasibility.pdf \
  output/pdf/jewels_neurips2026_concept_feasibility.pdf
latexmk -cd -c -jobname=jewels_neurips2026_concept_feasibility \
  paper/neurips2026/main.tex
```

The reviewed deliverable is copied to
`output/pdf/jewels_neurips2026_concept_feasibility.pdf`.

## Claim discipline

- Exact-compiler results and learned-speaker results are always separated.
- The 18-program macro dictionary is always described as source-backed.
- Failure of the learned speaker's strict rendered OpenCLIP top-1 gate is reported explicitly.
- Negative experiments localize unproven configurations; they are not treated as impossibility
  results.
- Manuscript prose states measurements and scope directly; official checklist guidance remains
  unmodified.
