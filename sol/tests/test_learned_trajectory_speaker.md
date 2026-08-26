# `test_learned_trajectory_speaker.py`

## Purpose

Verifies the learned speaker's autoregressive shapes, loss, and syntactically valid seeded sampling.

## Coverage

- scene/foreground/background logit shapes;
- finite decomposed cross-entropy;
- deterministic top-k sampling with distinct donors;
- invalid text shapes.
