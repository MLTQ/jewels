# `test_audit_coherent_source_oracle.py`

## Purpose

Checks that the coherent-source audit module imports and exposes the frozen entry point without
running the GPU evidence path.

## Coverage

- import-safe `main` entry point;
- protocol source is present beside the result family.
