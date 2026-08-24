# Residual appearance continuation v1 — frozen protocol

## Trigger

The registered 600-step residual experiment passed its representation rule before this continuation
was chosen. Residual-control reached exact `19.35552 dB / 0.74755` LPIPS and raw-response reached
`19.37601 / 0.74696`, versus bounded control `18.50932 / 0.75795` and bounded midpoint
`18.50301 / 0.75308`. Both residual arms passed structure. Raw-response narrowly improved both
macro metrics over residual-control, but per-style deltas were heterogeneous.

## Registered continuation

- Continue `screens/control_seed0_600/encoder.pt` and `screens/response_seed0_600/encoder.pt` for
  exactly 600 additional steps each (reported as total step 1,200).
- Preserve each arm's objective exactly: residual-control has no responsibility loss; raw-response
  retains raw RGB/Jacobian weights `0.025 / 0.025`, starting after step 100 and reaching full weight
  at step 400 of the new tranche.
- Preserve source corpus, fitted teachers, seed 0, AdamW LR `1e-4`, warmup 100, render/student sample
  counts, all structural weights, support settings, and five held-out validation sources.
- Resetting optimizer/schedule is shared by both arms and matches the project's earlier declared
  continuation semantics. No loss weight or appearance parameterization changes are authorized.

## Decision rules

Both arms must retain occupancy `<=0.985`, active fraction `<=0.70`, mixed tilt `>=0.25`, and finite
training. Every eligible arm is exact-audited together.

The continuation is useful only if at least one residual arm improves exact PSNR or LPIPS over its
own step-600 state without losing the other metric, and still beats both bounded references on both
metrics. Raw responsibility remains a causal pass only if response beats the equally continued
residual-control on both exact macro metrics; per-style coverage is reported and cannot be replaced
by the macro sign.

Absolute promotion remains `>=20 dB / <=0.40 LPIPS`. Contact sheets must be checked specifically for
contrast/boundary recovery and negative/dark speckling enabled by the unconstrained residual. After
this audit, unchanged seed-1/2 replication is justified for a retained joint representation result;
it is not contingent on raw response winning every style.
