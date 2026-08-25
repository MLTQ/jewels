# Frozen-base native local appearance adapter v1

> **Convergence correction (2026-08-25):** this report faithfully records a 400-update screen but
> generalized that short screen too far. A subsequent preregistered convergence study trained the
> same arms to 12k–16k updates. The derivative-x32 arm crossed the `LPIPS < 0.70` gate by 3.2k,
> reached `0.65895 LPIPS / 20.83538 dB` at 12k, and independently replicated at
> `0.65953 / 20.83497`. The recommendation below not to train longer is retracted. See
> [`../local_adapter_convergence_v1/README.md`](../local_adapter_convergence_v1/README.md).

## Outcome

At 400 updates, direct train-only LPIPS supervision improves the frozen irregular Jewelfield
without moving its geometry or rewriting its proven appearance base, but the tested per-jewel
color/Jacobian adapters have **not yet** reached the registered `LPIPS < 0.70` pitch gate or visibly
recovered sharp boundaries.

The strongest LPIPS arm reaches exact `20.08715 dB / 0.716935 LPIPS`, versus the frozen source at
`20.08200 / 0.722011`. The forced-evidence derivative arm reaches the best adapter PSNR,
`20.11852 / 0.717451`, and is mathematically unable to use generic capacity at zero radius. All 20
geometry tensors and all 40 non-adapter parameter tensors remain bitwise source-equal.

This is a useful mechanism result, not a selection result. It proves that direct perceptual
supervision can move the correct held-out metric while preserving the time-distorted irregular
field and 20 dB reconstruction. It also localizes the next bottleneck: changing only each existing
Gaussian's color and color Jacobian does not supply enough local spatial support for sharp detail.

## Registered source and gate

Every arm starts from the replicated seed-0 frozen residual checkpoint at exact
`20.08200 dB / 0.722011 LPIPS`. Geometry, residual appearance, and background are frozen; only a
zero-output adapter is optimized. Training uses the same 120 training clips and five held-out
cooking/style sources as the prior milestone.

The registered positive gate was:

- exact macro PSNR at least `20 dB`;
- exact LPIPS below `0.70`, with every seed and at least four of five styles improved;
- bitwise-exact geometry and base ownership;
- no worse range/temporal behavior and visibly sharper boundaries without speckle collapse.

No arm crossed `0.70`, so no fresh-seed replication was launched. The selection set has now been
observed; any future architecture winner needs replication and preferably a fresh held-out set.

## Stage 1: native raw samples do not cause the render gain

The raw adapter receives the existing stride-2 fine/coarse features plus a native-resolution
7-point RGB cross at each detached irregular jewel center. Radius zero collapses all seven samples
to the center while preserving the same parameter count.

| 400-update arm | Exact PSNR | Exact LPIPS |
|---|---:|---:|
| frozen source | 20.08200 | 0.722011 |
| radius-0, render MSE | 20.11216 | 0.720329 |
| radius-2 / time-1, render MSE | 20.11220 | 0.720316 |
| radius-0, LPIPS 0.01 | 20.11233 | **0.718634** |
| radius-2 / time-1, LPIPS 0.01 | 20.11233 | 0.718660 |

Radius-2 minus radius-0 is `+0.000033 dB / -0.000013 LPIPS` under render MSE and
`0.000000 dB / +0.000026 LPIPS` under LPIPS training. These are operationally zero and have
opposite LPIPS signs. Extra generic residual capacity and the objective—not raw local evidence—cause
the gain in this parameterization.

## Stage 2: direct perceptual pressure is a real positive mechanism

Read-only one-step calibration measured the 0.01 LPIPS term at `0.00713` versus sampled render MSE
`0.01220`; gradient norm changed only `0.00834 -> 0.00859`. The loss was evaluated on deterministic
train-only `2 x 96 x 144` frames every four updates with frequency correction. LPIPS network weights
were frozen.

| Raw adapter LPIPS weight | Exact PSNR | Exact LPIPS | Sampled RGB out of range |
|---:|---:|---:|---:|
| 0.01 | **20.11233** | 0.718660 | 3.193% |
| 0.05 | 20.08715 | **0.716935** | 3.355% |

Both exact metrics improve from source at weight 0.05. The stronger objective supplies an additional
`0.00173` LPIPS improvement over 0.01 while spending `0.02518 dB`. It remains visually subtle and
does not cross `0.70`, so this is evidence for direct perceptual optimization—not evidence that the
current adapter is sufficient.

## Stage 3: forced local evidence and gradient calibration

The derivative contract removes generic inputs and all biases. Its only inputs are central RGB
differences in x/y/time and six-neighbor contrast. At radius zero these features and its output are
exactly zero even with nonzero weights, so any learned effect depends on native irregular local
evidence.

The first unscaled screen measured gradients about 30 times smaller than the stable raw adapter and
moved LPIPS only slightly (`20.08431 / 0.721649` at weight 0.05). A preregistered fixed input scale
32 matched the observed gradient regime and is explicit in checkpoint metadata.

| Forced derivative arm | Exact PSNR | Exact LPIPS | Final residual Jacobian energy |
|---|---:|---:|---:|
| scale 1, LPIPS 0.05 | 20.08431 | 0.721649 | 0.05239 |
| scale 32, LPIPS 0.05 | **20.11852** | **0.717451** | 0.10267 |

Scale 32 is a real improvement over source and keeps sampled out-of-range RGB at `3.091%`, but it
misses the preregistered requirement to beat the raw 0.05 LPIPS result by `0.000516`. Its doubled
Jacobian energy also argues against extending it without a better spatial basis. The arm is not
selected.

## Visual and structural audit

`audit_final_seed0_400/qualitative.png` shows, left to right, target, lattice, frozen source,
radius-0 LPIPS 0.01, raw local LPIPS 0.05, derivative-x32 LPIPS 0.05, and fitted teacher. Differences
among irregular arms are subtle; none restores the subject/object boundaries visible in the lattice
or teacher, and none introduces a new collapse.

`audit_final_seed0_400/field_layout.png` confirms that every adapter retains the same irregular XY
and X-time center field. `geometry_exact_final.json` independently compares all 20 geometry
checkpoint tensors with exact equality and zero maximum change. Every training summary reports all
40 frozen non-adapter parameter tensors bitwise exact.

`evidence.png` separates exact frontier metrics, the radius causal controls, objective/feature
progression, and sampled range/Jacobian diagnostics. `evidence.json` contains its numbers.

## Decision and next gate

The original instruction not to train these arms longer was disproved by the convergence study and
is withdrawn. The fitted ceiling (`27.699 dB / 0.172 LPIPS`) and lattice (`23.652 / 0.428`) still show
that the video and renderer contain far more recoverable detail, but the derivative adapter itself
was undertrained at 400 updates and remains the selected appearance mechanism.

Child splats remain a possible later capacity experiment, not the immediate conclusion of this
screen. The selected derivative adapter should first be tested on a fresh selection set, a third
training seed, and explicit RGB-range guardrails; then its encoder/data scaling curve should be
measured before adding representation capacity.

A useful next gate is exact PSNR at least `20`, LPIPS below `0.70`, improvement in at least four of
five styles, and visible boundary/detail recovery. Radius-zero/no-child and shuffled-patch controls
must distinguish new local support from generic capacity.
