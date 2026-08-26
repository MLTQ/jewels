# `test_train_learned_trajectory_speaker.py`

## Purpose

Verifies the frozen program-pair split and correct/shuffled/null evaluation logic without running
OpenCLIP or GPU optimization.

## Coverage

- cyclic donor pairs are held out and train/evaluation pairs are disjoint;
- three prompts and six donors per scene produce expected row counts;
- condition evaluator returns finite token metrics.
