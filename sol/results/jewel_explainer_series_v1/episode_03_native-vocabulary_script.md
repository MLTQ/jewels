# Episode 3: Turning Jewels into a native language

Gauge-free features, factor tokens, and continuous positions

## Claim sources

- `stprim/prior/featurize.py`
- `sol/factorized_jewel_casting_language.py`
- `sol/prompt_jewel_caster.py`
- `sol/results/jewel_casting_language_v0/hierarchical_v1/gate0f_individual/report.json`

## 1. Why tokenization is nontrivial

A language needs recurring symbols, but fitted Gaussian parameters are not naturally unique. Two optimization runs can describe the same visual field with different row order, slightly shifted centers, and different rotation parameterizations. Quantizing raw parameter tuples would spend vocabulary capacity on accidental coordinate choices. The first job is therefore not compression. It is to define a stable physical alphabet whose tokens preserve rendering and spacetime structure.

**On screen:** The target is a stable physical alphabet—not a codec bit rate.

## 2. Rotation has gauge ambiguity

A unit quaternion q and its negation represent exactly the same rotation. Even after choosing quaternion sign, covariance eigenvectors may permute when their eigenvalues are reordered, and each eigenvector admits a sign flip. Feeding scale plus quaternion to a prior makes identical ellipsoids appear far apart. This is pure representational noise: it changes parameter coordinates without changing a single rendered pixel.

**On screen:** q and −q render identically; covariance axes also permute and flip.

## 3. Log-covariance removes the gauge

The canonical representation stores the matrix logarithm of the symmetric positive-definite covariance. A symmetric three-by-three matrix needs six unique numbers. The logarithm maps multiplicative scale differences into a well-behaved Euclidean coordinate chart. Decoding exponentiates, eigendecomposes, and converts one valid rotation back to the renderer. Any sign or axis choice made during that last factorization renders the same covariance.

**On screen:** log Σ is unique and symmetric; six numbers replace scale-plus-quaternion gauge.

## 4. One joint token was too rigid

The first vocabulary treated an eight-Jewel constellation as one joint one-hundred-and-seventy-six-dimensional prototype. That couples layout, covariance, surface color, and color gradient into one indivisible choice. Gate zero-a falsified that design. The successful language assigns independent codebooks to physical roles, then composes their aligned prototypes back into the same constellation. Factorization lets reusable geometry combine with reusable appearance.

**On screen:** Reject one 176-D joint token; compose independent physical roles.

## 5. Combinatorial capacity

Four one-thousand-and-twenty-four-way decisions expose up to one-thousand-and-twenty-four to the fourth role combinations without fitting that impossibly large joint table. For individual generated Jewels, the centroid remains continuous and the three nonconstant roles—covariance, surface, and gradient—are active tokens. Layout becomes the spoken centroid itself. The codebooks share normalization and preserve the canonical twenty-two-feature contract.

**On screen:** 4 codebooks × K=1024 expose compositional capacity without a K^4 table.

## 6. Addresses are internal; centers are continuous

Cells are useful for learning local histograms and addressed phrases, but an address is not an emitted coordinate. The generator outputs centroids in continuous normalized space. Internal cell indices answer questions like which local token distribution applies here. They do not snap a Jewel to the cell center. This separation directly addresses the grid-quantization artifact: routing may be discrete while geometry remains irregular.

**On screen:** discrete routing cell ≠ emitted centroid; μ stays continuous.

## 7. Gate zero-f: the physical alphabet survives

At the selected individual-Jewel language gate, decoded irregular fields retain twenty-two-point-eight-six-five-seven decibels against the continuous source on the frozen random-volume audit, preserve mixed spacetime tilt at one-point-zero-four-one-five times the source, and show zero center locking. These numbers do not prove promptability. They prove that the physical token vocabulary is not the bottleneck preventing a higher-level speaker from producing renderable continuous fields.

**On screen:** 22.8657 dB · 1.0415× tilt retention · 0% grid locking
