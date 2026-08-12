# Scaffold topology and autonomous continuation gate

This folder records the first held-out LTX continuation in which fitted birth cells, counts, and
ranks are replaced by predictions from the prompt-generated video scaffold. The 0.88M-parameter
topology head trains on 12 UCF fitted fields. The selected UCF-trained 22-D mark flow remains frozen.
Every learned rank is synthesized, projected into its predicted cell, and merged after unchanged
carried jewels.

## Result

The topology head establishes causal scaffold use on all twelve LTX initial/continuation views:

| Topology input | Slot F1 | Count correlation | Birth-count ratio |
|---|---:|---:|---:|
| Correct scaffold | **0.6636** | **0.6010** | 0.9390 |
| Cross-class shuffled scaffold | 0.6092 | 0.4728 | 0.9341 |
| Null scaffold | 0.6091 | 0.4628 | 0.8829 |
| Per-stride UCF train mean | 0.6222 | 0.3147 | 0.9378 |

The 3,000-update run takes 41.96 seconds on the RTX 2070 Super. Training slot F1 is 0.9424,
revealing a domain/generalization gap, but the correct held-out scaffold beats every causal and
constant control. Omitting carry is decisively harmful: slot F1 falls to 0.5374 and the count ratio
rises to 1.193.

An apparent sequential-density failure in the oracle-mark diagnostic is not a valid generation
failure. That diagnostic retains only fitted marks whose exact cell/rank also exists in the
prediction, producing 56--59% of fitted effective density. Gaussian decompositions are non-unique:
a predicted rank that does not match one particular fitted rank still needs a newly synthesized
mark. Two source-level LTX cross-validation folds confirm that adding two in-domain fits improves
topology discrimination but does not repair this exact-rank-retention measure, so a larger fitted
corpus is not the immediate remedy.

## Frozen-realizer coupling

The decisive gate synthesizes every predicted rank for the 32--48 continuation on each held-out
LTX clip. The correct LTX scaffold conditions topology; the frozen mark flow receives the correct
future raster in every learned/shuffled topology panel so that the topology intervention is
isolated.

| Panel | PSNR | SSIM | Contrast | Edge | Effective jewels/frame |
|---|---:|---:|---:|---:|---:|
| Carried only | 13.529 | 0.3521 | 0.3263 | 0.3438 | 1,924 |
| Oracle topology + generated marks | 15.492 | 0.6997 | 0.7252 | 0.8197 | 6,252 |
| **Learned topology + generated marks** | **15.464** | **0.6967** | **0.7202** | **0.8250** | **5,822** |
| Shuffled topology + correct mark guide | 15.353 | 0.6927 | 0.7314 | 0.8170 | 5,781 |

Learned topology predicts 70,724 births versus 70,952 target births (99.68%). Its continuation-only
slot F1 is 0.6877 versus 0.6505 shuffled and 0.6589 null; count correlation is 0.5960 versus
0.4859/0.4989. All four learned fields average 5,598--6,449 effective contributors/frame, inside
the intended 5k--10k regime. The largest learned continuation cell has 113 ranks, safely below the
frozen realizer's 512-rank capacity. Spatial projection adherence is 100%, birth-cell adherence is
98.85--99.27%, and the maximum carried-feature error is exactly 0.0 for every source.

Learned topology loses only 0.028 dB and 0.0030 SSIM to oracle topology. Visual review agrees: the
learned and oracle columns preserve the same macro-layout and share the same motion-region noise.
The shuffled topology panel is only modestly worse because the coarse grid is nearly fully occupied
and the correct video guide still supplies semantics to the mark flow. The next quality bottleneck
is therefore foreground/motion mark realization and stronger spatial count selectivity, not raw
birth density or a longer run of the same topology model.

## Artifacts

- `topology_summary.json` and `topology_train_log.jsonl`: leakage-safe UCF-train/LTX-validation
  topology run.
- `cv_fold_a_summary.json` and `cv_fold_b_summary.json`: diagnostic two-fold LTX source-level
  cross-validation; these do not replace the untouched four-source gate.
- `realizer_summary.json`: authoritative four-source learned-topology/frozen-realizer report.
- `autonomous_topology_contact.png`: one middle continuation frame per class. Columns are fitted
  target, carried only, oracle topology, learned topology, and shuffled topology.
- `*_autonomous_topology.gif` and `*_contact.png`: all 16 continuation frames and three-time visual
  controls for each class.

The retained remote checkpoints are:

- topology: `/home/m/jewels/topology/scaffold_topology_v1/run_3000/scaffold_topology.pt`, SHA-256
  `f210820aa88738bd6d8146f6f063e514dd9a310ffcabad56c0d2d6bcd7d8362a`;
- frozen mark flow: `/home/m/jewels/tokenizer/prompted_mark_flow_oracle_guide_12000/prompted_mark_flow.pt`,
  SHA-256 `70fd301be096e361c9f05bb6432b24d7ef01676de680a63833e21e292e8f3e52`.

This passes autonomous topology and density for one continuation stride. It does not yet generate
the initial jewel state or prove two free-running generated-mark strides.
