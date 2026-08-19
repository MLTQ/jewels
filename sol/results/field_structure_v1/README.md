# What the lattice encoder discards: fitted vs encoder field structure

Path B's target measurement. The fitter produces content-adaptive fields; the amortized encoder
produces a lattice that samples the video. This quantifies the gap on two LTX training clips
(72k fitted jewels vs 73,728 encoder jewels, same videos).

| metric | fitted | encoder | meaning |
|---|---:|---:|---|
| anisotropy median | **10.25** | **2.21** | fitted jewels are sheared tubes; encoder jewels are near-round |
| anisotropy p90 | 29.88 | 2.41 | fitted has extreme tubes; encoder has none |
| extent IQR ratio | 2.28 | 1.07 | fitted varies size with content; encoder is uniform |
| occupancy uniformity | 0.946 | 0.9992 | fitted clusters on content; encoder is a near-perfect grid |
| opacity median | 0.286 | 0.434 | encoder leans on many mid-opacity blobs |
| temporal tilt median | 0.948 | 0.998 | **not interpretable** — see caveat |

**Caveat on temporal tilt:** with near-isotropic covariance the principal axis is arbitrary, so
the encoder's 0.998 is not evidence its blobs track motion. Only the fitted number carries
meaning, and even there it should be read alongside anisotropy.

## Reading

The founding premise of the project is that a moving object is *one* sheared spacetime tube
rather than a new primitive per frame. The fitter delivers that (median anisotropy 10.25,
p90 29.9). The encoder does not (2.21, p90 2.41): it lays down uniform round blobs on a grid
and colors them by sampling the video, which is why its learned features contribute only
0.15 dB at inference and why quantizing them looked free.

## Target for a structural encoder

A Path B encoder should reach approximately:

- anisotropy median >= ~10 (tubes, not blobs)
- extent IQR ratio >= ~2 (size follows content)
- occupancy uniformity <= ~0.95 (clustering, not lattice)

while reconstructing competitively — ideally at far fewer jewels, since descriptive primitives
should not need 73k per window.

**Known failure mode to guard against:** uniform blobs are the safe local optimum under a plain
reconstruction loss, so a feed-forward model will drift back to the lattice unless the design
prevents it (fewer primitives, no color copying, a real bottleneck).

Artifacts: `report.json` (per-clip and macro).
