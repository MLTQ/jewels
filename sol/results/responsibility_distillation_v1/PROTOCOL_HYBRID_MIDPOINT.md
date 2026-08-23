# Hybrid appearance midpoint v1 — frozen protocol

## Question

Do the exact responsibility-only and half-strength-local hybrid endpoints contain a joint PSNR/LPIPS
win at their single raw-local-weight midpoint?

## Evidence available before registration

Under the same source, seed, 600 steps, responsibility objective, and matched control:

| Arm | Local RGB / gradient | Exact PSNR delta | Exact LPIPS improvement |
|---|---:|---:|---:|
| responsibility appearance | `0 / 0` | `+0.05865 dB` | `-0.00018` |
| half-strength local hybrid | `0.05 / 0.10` | `-0.03261 dB` | `+0.00364` |

The midpoint is selected once from these two completed endpoints. Linear interpolation would place it
near `+0.013 dB` PSNR and `+0.00173` LPIPS improvement, but this is only a falsifiable motivation;
neural optimization is not assumed linear. No further weight interpolation is authorized by this
protocol.

## Registered midpoint

- All source, data, seed, optimizer, 600-step compute, global structural losses, validation styles,
  sampling, support, and schedules remain identical to `PROTOCOL_HYBRID.md`.
- Responsibility RGB/Jacobian remain `0.025 / 0.025`.
- Position-only local RGB/gradient are exactly `0.025 / 0.05`, the midpoint between the two audited
  endpoints. All local/responsibility geometry and opacity weights remain zero.
- The already completed `screens/control_seed0_600` is the matched control. Independent teacher RNGs
  preserve its descriptor samples and GPU training sequence.

## Decision rule

Sampled eligibility remains occupancy `<=0.985`, active fraction `<=0.70`, mixed tilt `>=0.25`, and
PSNR no more than `0.50 dB` below control. If eligible, the midpoint is exact-audited beside control.

The midpoint passes the mechanism gate only if exact macro PSNR is greater and LPIPS is lower than
control with all structural gates retained. Absolute promotion remains `>=20 dB` and `<=0.40`
LPIPS. A mechanism pass below the absolute gate warrants a longer/data-scale confirmation; failure
ends this interpolation tranche and redirects work to the bounded appearance contract.
