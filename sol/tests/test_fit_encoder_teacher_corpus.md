# `test_fit_encoder_teacher_corpus.py`

## Purpose

Protects leakage-safe, ordered source selection and portable checkpoint identity for the
support-correct teacher corpus.

## Components

### `TeacherCorpusTests`
- **Does**: Verifies prefix/slice and explicit-order selection, missing-source rejection, and
  filename sanitization.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Teacher fitter | Balanced manifest order is preserved | Selection order |
| Distillation loader | One source ID maps to one portable file | Naming policy |
