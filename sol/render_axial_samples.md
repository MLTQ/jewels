# `render_axial_samples.py`

## Purpose

Turns hierarchical-prior distribution metrics into inspectable 45k-jewel videos on unseen source
conditions.

## Components

### `main`
- **Does**: Samples normalized 16³ codes from the axial EMA prior, restores PCA/fine tokenizer
  scales, sparse-decodes jewels, and writes fitted/hierarchy/generated comparison GIFs.
- **Interacts with**: Both latent caches, `block_codec.py`, architecture-aware checkpoint restorers,
  `latent_prior.sample_flow`, and production render helpers.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Visual generation gate | Panel order is target / deterministic hierarchy / axial sample | Visual semantics |
| Hierarchical decode | Coarse normalization is undone before inverse PCA; fine normalization after | Scale order |
| Research record | Manifest carries counts, source, frame picks, and artifact | Manifest schema |

## Notes

- CFG defaults to 1.0 because the dense six-scene run did not establish positive conditioning gain.
