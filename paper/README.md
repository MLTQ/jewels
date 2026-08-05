# Jewels paper

The editable source for the technical progress report is `main.tex`; citations live in
`references.bib`. The report intentionally rejects the broad claim that Jewels is the first use of
Gaussian splats to reconstruct video and instead documents the narrower generative/editable
research hypothesis.

Build from the repository root:

```bash
mkdir -p tmp/pdfs output/pdf
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=jewels_progress_report -outdir=../tmp/pdfs paper/main.tex
cp tmp/pdfs/jewels_progress_report.pdf output/pdf/jewels_progress_report.pdf
```

Render for review:

```bash
pdftoppm -png -r 120 output/pdf/jewels_progress_report.pdf \
  tmp/pdfs/jewels_progress_report
```
