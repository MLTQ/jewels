# Path B v1: can a feed-forward encoder produce descriptive jewels?

Two controlled arms of the same 10,240-jewel structural encoder (2,048 cells x 5 slots, 5.09M
params, quaternion+scale shape, no video-colour lookup, positions free to migrate +/-2 cells).
They differ only in whether the fitter's structure is supervised.

Reference points: the **lattice encoder** (73,728 jewels, samples video colours) and the
**fitter** (72,000 jewels, 9,000 optimization steps per clip).

## Result

| metric | lattice encoder | render-only | **distilled** | fitter |
|---|---:|---:|---:|---:|
| anisotropy median | 2.21 | 4.28 | **6.29** | 10.25 |
| extent IQR ratio | 1.07 | 1.31 | **1.84** | 2.28 |
| occupancy uniformity | 0.9992 | 0.9994 | **0.9909** | 0.946 |
| held-out PSNR | 23.4 (73k jewels) | 19.18 (180 windows) | 15.96 (12 windows) | — |

**PSNR is not comparable across arms** — the distilled run trains on the 12 windows that have
fitted fields, the render-only run on 180. Only the step-matched structural comparison is
meaningful.

## The mechanism: L2 actively prefers a lattice

Render loss alone degrades structure monotonically as it improves quality:

| step | PSNR | anisotropy | extent IQR | uniformity |
|---:|---:|---:|---:|---:|
| 1000 | 14.33 | 3.22 | **2.10** | 0.9961 |
| 4000 | 18.11 | 4.27 | 1.40 | 0.9986 |
| 8000 | 19.18 | 4.28 | **1.31** | **0.9994** |

Extent variation *peaks at step 1000* — near the fitter's 2.28 — and optimization destroys it.
Uniformity rises monotonically to 0.9994, marginally worse than the lattice encoder it was
designed to beat. The reason is structural: L2 rewards adequate coverage everywhere, so
clustering onto detail starves flat regions and raises total error. Uniform tiling is the
optimum and gradient descent finds it reliably. The fitter escapes only because densify/prune
is an explicit mechanism *outside* the loss.

## Distillation breaks the anticorrelation

Adding symmetric Chamfer on positions plus scale-invariant anisotropy-spread matching against
fitted fields:

| step | PSNR | anisotropy | extent IQR | uniformity |
|---:|---:|---:|---:|---:|
| 1000 | 15.83 | 6.48 | 1.81 | 0.9952 |
| 3000 | 16.05 | 6.52 | 1.73 | 0.9922 |
| 4000 | 16.40 | 6.38 | 1.75 | 0.9910 |
| 6000 | 15.96 | 6.29 | 1.84 | **0.9909** |

Quality rises while structure holds or improves — the opposite of the render-only arm, and the
first run in this project where occupancy uniformity moves *toward* the fitter.

## Reading

1. **Feed-forward content-adaptive fields are achievable.** A single forward pass reaches 61% of
   the fitter's anisotropy and 81% of its extent variation, against 9,000 optimization steps per
   clip. Shape adaptation is largely solved.
2. **The obstacle was the objective, not the architecture.** Scarcity and an expressive shape
   parameterization were necessary but not sufficient; without structural supervision the same
   network collapses to a lattice.
3. **Placement remains only partly solved.** Uniformity 0.9909 vs the fitter's 0.946: jewels
   adapt their shape to content far better than their position. Leading hypothesis — the teacher
   subsample (12,000) is nearly 1:1 with the student's 10,240, so covering every teacher jewel is
   satisfiable without migration. A sparser teacher target, or freer position bounds, tests this
   directly and is the obvious next experiment.
4. Only 16 windows have fitted fields; the 240-clip diverse corpus is unfitted, which caps how
   far distillation can be pushed until more fits exist (~25 GPU-min each).

## Consequence for jewels-as-tokens

The lattice encoder's field was a re-encoded video (learned features worth 0.15 dB, quantization
free because the quantized half carried nothing). This field is materially more descriptive, so
it is a better token substrate — but the token claim should be re-tested on *these* fields, not
assumed from the structure metrics.

Artifacts: `render_only_summary.json`, `render_only_train_log.jsonl`, `distilled_summary.json`,
`distilled_train_log.jsonl`.
