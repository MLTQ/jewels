# `test_coherent_source_realizer.py`

## Purpose

Verifies that the coherent-source causal control chooses exactly one scene-eligible field for the
entire window and emits its complete active-token Jewel program.

## Coverage

- correct scene eligibility and global source selection;
- exact requested count and output shapes;
- rejection of invalid block-program shapes.
