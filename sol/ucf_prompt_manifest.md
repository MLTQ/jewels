# `ucf_prompt_manifest.py`

## Purpose

Builds the leakage-safe four-class UCF smoke manifest for prompt-conditioned streaming. It binds
source-group splits, prompt templates, frozen text-encoder identity, and the validated dense fitting
contract before expensive checkpoints are created.

## Components

### `PromptClassSpec` / `PROMPT_SPECS`

- **Does**: defines three training phrasings and an unseen evaluation phrasing for each smoke class
- **Rationale**: the model must learn class meaning rather than one fixed prompt string

### `parse_ucf_video` / `scan_candidates`

- **Does**: validates canonical UCF class/group/clip filenames and obtains exact frame counts with
  `ffprobe`

### `select_balanced_candidates`

- **Does**: chooses the longest clip meeting the 96-frame contract in every requested class/group
- **Rationale**: one clip per independent source group maximizes diversity within the fitting budget

### `stage_candidates`

- **Does**: creates an idempotent symlink directory that the existing resumable corpus fitter can
  consume without broad class globs

### `build_manifest`

- **Does**: serializes source identity, train/validation ownership, prompts, CLIP identity, and the
  120k spatial-split fit contract

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense fitter | `fit_video` paths are unique staged videos; one 96-frame window each | Path or frame fields |
| Prompt encoder | Prompt strings and exact OpenCLIP model/pretrained identity are immutable | Encoder metadata |
| Continuation trainer | Every class occurs in train and validation; source groups never cross | Split semantics |
| Evaluation | Evaluation templates are absent from training templates | Prompt lists |

## Notes

- The default classes are Basketball, HorseRiding, PlayingGuitar, and ApplyEyeMakeup. They provide
  distinct scene scale, motion, and appearance while all offering four eligible source groups.
- Group 4 is held out in every class. UCF group IDs denote independent source videos within a class.
