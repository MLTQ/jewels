# adapt.py

## Purpose
Adaptive primitive-count control. A fixed budget can't follow content, and in a spacetime volume
the dynamic range is extreme: a static background wants a handful of enormous primitives while a
moving limb wants hundreds of small ones.

## Components

### `GradientTracker`
- **Does**: accumulates per-primitive |d(loss)/d(mu)| between adaptation steps
- **Rationale**: positional gradient is the densification signal — a primitive that "wants" to be
  in several places at once should become several primitives.
- Self-heals if primitive count changes underneath it.

### `adapt(field, tracker, ...)`
- **Does**: prune low-weight primitives, then split the highest-gradient survivors
- **Interacts with**: `PrimitiveField.subset_`/`append_`, called by `fit/fitter.fit_volume`
- **Rationale**: prune-then-densify so the gradient ranking is computed over survivors. Parents
  are shrunk alongside children so total energy is roughly conserved across a split.

## Decisions
- Single split operation rather than 3DGS's separate clone/split. Simpler; the shrink factor
  covers both regimes adequately at this stage.
- **Caller must rebuild the optimizer.** Rejected surgical Adam-state splicing — it's a known
  source of silent bugs in 3DGS implementations and the momentum loss is tolerable at these
  adaptation intervals.
- Never prunes below 8 primitives.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `fit/fitter.py` | returns stats dict; optimizer invalid afterwards | The rebuild contract — silent corruption if violated |

## Notes
- Densification stops at `adapt_until_frac` of training so the final stretch is pure refinement.
