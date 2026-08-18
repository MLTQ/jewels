# `evaluate_latent_text_prior.py`

## Purpose

The authoritative G3 test: samples the latent prior under correct and shuffled prompts with an
identical noise seed, decodes both through the frozen encoder, renders them, and scores against
the target video. Latent-space MSE cannot settle text selectivity when unpredictable detail
dominates the latent; rendered output can.

## Components

### `render_latent`
- **Does**: Denormalizes a generated latent, unpacks it to `cells`/`seed`, decodes to jewels
  through the frozen encoder, and renders on a uniform `(u,v,t)` grid.

### `main`
- **Does**: For each held-out window, generates the correct/shuffled pair, renders both, and
  reports render and texture-blind layout signatures plus the macro correct-minus-shuffled gap.
- **Selection**: Held-out windows are drawn round-robin across styles, not in manifest order —
  otherwise the sample collapses onto one domain and never tests the style axis, which the
  latent-variance analysis identifies as the strongest text signal.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Stage 1 G3 decision | Schema `latent-text-prior-render-gate-v1` with paired arms | Report fields |

## Notes

- Pairs share a noise seed, so any difference is attributable to conditioning alone.
