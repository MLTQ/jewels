# Single-field memorization renderer tests

These tests cover the evaluator's experiment-identity guard and teacher-forced
density stitching. The identity guard accepts exactly one train/validation alias
pair that shares one physical fitted field and rejects a pair backed by different
fields. Density stitching retains only each window's committed stride rather
than measuring the entire target field repeatedly.

Full checkpoint sampling and GIF production are exercised by the Aine experiment
because they require the fitted cel corpus and a CUDA renderer.
