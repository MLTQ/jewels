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

### `adapt(field, tracker, ..., generator)`
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
- `split_mode="isotropic"` preserves the historical three-axis shrink and exact checkpoint
  behavior. `split_mode="spatial"` retains the principal axis most aligned with time, shrinks the
  other axes by √2, and rotates local split jitter into world coordinates. Two children therefore
  conserve spatial footprint without shortening their temporal lifespan.
- When the fitter supplies its CPU generator, split jitter is drawn on CPU and transferred to the
  field device. This puts sampling and adaptation on one serializable random stream for exact
  restart behavior across accelerators. Callers omitting `generator` retain device-local random
  jitter.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `fit/fitter.py` | returns stats dict; optimizer invalid afterwards | The rebuild contract — silent corruption if violated |

## Notes
- Densification stops at `adapt_until_frac` of training so the final stretch is pure refinement.
- The spatial mode is the research path for spacetime fields. The legacy default remains isotropic
  until its density/quality control fit passes.
