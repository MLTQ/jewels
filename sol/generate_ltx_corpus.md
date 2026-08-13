# `generate_ltx_corpus.py`

## Purpose

Generates the first balanced prompt-to-video scaffold corpus for Jewels using the installed LTX-2.3
pipeline. It turns the established UCF prompt split into resumable videos and an auditable corpus
manifest without changing the source prompt contract.

## Components

### `CorpusExample` / `build_plan`

- **Does**: Flattens three training and one evaluation phrasing per class into stable sample names,
  roles, prompts, and deterministic seeds.
- **Interacts with**: `classes` in `results/prompt_smoke/manifest.json`.
- **Rationale**: A 16-sample batch is the smallest balanced LTX gate matching the existing four-class
  text-conditioning split.

### `select_plan`

- **Does**: Selects all prompts or only the train/evaluation role while retaining the source
  manifest's class order, seeds, stems, and prompt identities.
- **Rationale**: A role filter can produce a balanced four-class evaluation gate; truncating the
  class-major 16-sample plan cannot.

### `CorpusRuntime` / `scaffold_config`

- **Does**: Binds every sample to one resolution, frame count, model revision path, GPU UUID,
  quantization, and offload policy.
- **Interacts with**: `ScaffoldConfig` in `ltx_scaffold.py`.

### `write_corpus_manifest`

- **Does**: Atomically records source-manifest digest, prompt roles, completion state, receipt paths,
  runtime, peak GPU memory, and media probes.
- **Rationale**: An interrupted service can be relaunched without losing provenance or duplicating
  valid clips.

### `run_corpus`

- **Does**: Generates missing examples sequentially and refreshes the manifest after each attempt.
- **Interacts with**: `run_scaffold` in `ltx_scaffold.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Video-to-jewel corpus loader | One MP4/receipt pair per manifest entry | Entry/output schema |
| Prompt evaluation | Training and evaluation phrasings remain explicitly labeled | Role semantics |
| Experiment audit | Source manifest SHA-256 and deterministic seeds remain recorded | Digest/seed policy |
| Resume service | Completed matching receipts are skipped; failed/missing samples rerun | Match criteria |
| Styled evaluation gate | `--prompt-role evaluation` returns one held-out prompt per class | Role filtering/order |

## Aine run

The v1 corpus uses 16 clips at `768x512`, 49 frames, 24 fps, FP8 cast, CPU offload, and RTX 4090
UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. Each semantic prompt receives the same neutral
cinematography suffix: realistic, continuous, stable, natural motion, and no cuts. The untouched
source phrasing remains separately recorded.

The run completed 16/16 clips with no failures in 46.13 aggregate minutes. Mean sample runtime was
172.98 seconds and aggregate peak GPU memory was tightly bounded at 9,398--9,401 MiB. All media
contracts and four class-level contact-sheet audits pass; results are recorded in
`results/ltx_scaffold_v1/README.md`.
