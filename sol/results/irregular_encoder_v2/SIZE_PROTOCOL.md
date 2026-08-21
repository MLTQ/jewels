# Factorized v3 absolute-size continuation — frozen protocol

## Motivation

The selected 40,960-proposal v3 arm passes geometry allocation/tilt gates but renders broad blur.
On the exact five-style audit its median extent is `0.05158`; the fitted teacher median is `0.01285`
(4.01× smaller), while median anisotropy is already close (`2.13` versus `2.43`). The original
spread loss cannot see this difference because it is scale-invariant.

## Matched causal test

- Source: the same `v3_slots20_seed0_600` checkpoint selected by `PROTOCOL.md`.
- Two 600-step continuations, seed 0, identical data/order/render/structure losses.
- Optimizer: AdamW continuation at `1e-4` peak learning rate with 100-step warmup.
- Control: absolute-size weight `0`.
- Intervention: smooth-L1 mean-log-scale weight `0.05`.
- Teacher size offset: `+0.35` log units. This permits a 1.42× larger linear extent than a 72k
  teacher because the selected field has about 25.6k active jewels; it remains far below the
  observed 4.01× excess.
- The appearance-grid objective remains disabled so the only changed gradient is absolute size.

## Selection

The intervention is successful only if median extent moves materially toward the adjusted target,
all three structural screen gates remain satisfied, and held-out PSNR does not fall by more than
0.25 dB versus control. A failed result narrows only this weight/offset/600-step continuation.

## Lower-weight bracket

The matched intervention moved median extent from `0.05093` to `0.02636` (48.3% smaller) and kept
all structural gates, but its `17.7263 dB` held-out PSNR was `0.5859 dB` below the `18.3122 dB`
control. This rejects weight `0.05`; it does not reject absolute-size supervision.

Before running them, three lower weights are registered: `0.01`, `0.02`, and `0.03`. Each restarts
from the same `v3_slots20_seed0_600` checkpoint and retains every control setting above. An arm is a
useful Pareto point only if:

1. median extent is at most `0.040` (at least 21% smaller than control);
2. held-out PSNR is at least `18.0622 dB` (within 0.25 dB of control);
3. occupancy is at most `0.985`, active fraction is at most `0.70`, and mixed tilt is at least
   `0.25`.

If multiple arms pass, select the highest-PSNR arm. If none passes, report the bracket as a scoped
failure and move to an adaptive size target or a longer joint appearance/geometry schedule rather
than changing the decision rule after seeing results.

## Boundary confirmation

None of the three bracket arms passed both limits. Weight `0.02` produced extent `0.04158` and
PSNR `18.1554 dB`; weight `0.03` produced extent `0.03622` and PSNR `18.0087 dB`. They smoothly
straddle the two unchanged decision boundaries. Linear interpolation predicts that weight `0.023`
will produce extent about `0.03997` and PSNR about `18.1114 dB`.

One confirmation at weight `0.023` is registered before execution, starting from the same source
and using the same 600-step settings. It passes only if it satisfies the existing `0.040` extent,
`18.0622 dB` PSNR, and three structural limits above. Passing makes it the candidate for an exact
audit; failure closes this fixed-offset, 600-step size experiment without another weight search.
