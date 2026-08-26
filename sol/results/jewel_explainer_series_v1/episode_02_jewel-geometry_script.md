# Episode 2: A Jewel is a Gaussian in spacetime

Geometry, appearance, and the exact additive renderer

## Claim sources

- `stprim/core/params.py`
- `stprim/prior/featurize.py`
- `stprim/models/render.py`
- `stprim/models/tiled_support.py`

## 1. Three coordinates, not four

A Jewel lives in normalized u, v, t coordinates: horizontal image position, vertical image position, and time. This is a three-dimensional spacetime volume. RGB is attached appearance, not a fourth Gaussian axis. To render frame t-zero, we slice the volume with a plane at that time and evaluate every pixel coordinate on the plane. Time distortion comes from an anisotropic covariance whose principal axes may tilt between space and time.

**On screen:** Jewel geometry lives in (u, v, t); a video frame is a time slice.

## 2. The twenty-two canonical features

The generative boundary represents each Jewel as twenty-two numbers. Three numbers store the centroid mu. Six store the upper triangle of the symmetric log-covariance. Three store constant RGB. Nine store the full three-by-three world-frame color Jacobian. One stores the weight logit. This layout is gauge-free: appearance gradients are expressed in world coordinates, and covariance is stored as a unique symmetric matrix rather than a rotation convention.

**On screen:** μ(3) | log Σ upper triangle(6) | RGB(3) | color Jacobian(9) | logit weight(1)

## 3. Covariance controls time distortion

Internally the renderer factorizes covariance into a proper rotation R and positive principal scales S. A Jewel elongated along u paints a horizontal region. One elongated along t persists across frames. A principal axis tilted jointly through u and t moves across the image as time advances. That mixed spacetime tilt is not post-hoc optical flow; it is literally encoded in the orientation of the Gaussian ellipsoid.

**On screen:** Sigma = R diag(s^2) R^T; tilted eigenvectors couple position and time.

## 4. Mahalanobis distance

For a query point x, displacement d equals x minus mu. The renderer rotates that displacement into the Jewel's principal frame with R-transpose, divides componentwise by scale, and sums the squares. This yields q, the squared Mahalanobis distance. Computing y equals S inverse R-transpose d avoids materializing an inverse covariance for every pixel-Jewel pair. The Gaussian contribution decays as exponential minus one-half q.

**On screen:** d = x - mu; y = S^-1 R^T d; q = ||y||^2

## 5. Appearance is locally linear

A constant-color Gaussian would need many small primitives to express a smooth color ramp. Instead, the default P-one appearance model stores a three-by-three Jacobian. Local color is base color plus that Jacobian times displacement from the centroid. One row describes how red changes with u, v, and t; the next rows do the same for green and blue. Temporal entries can therefore change a Jewel's color as a frame slice moves through it.

**On screen:** c_i(x) = c_i^0 + J_i(x - mu_i)

## 6. The field is additive

The final pixel is the learned constant background plus a sum of every supported Jewel contribution. Each weight is sigmoid of its stored logit, multiplied by the Gaussian exponential, multiplied by local color. There is no softmax normalization forcing Jewels to compete for ownership. An earlier soft-Voronoi model was implemented and steelmanned, but lost both reconstruction and canonicality, so the evidence path is explicitly additive.

**On screen:** value(x) = background + sum_i exp(-q_i/2) * sigmoid(a_i) * c_i(x)

## 7. Support-complete rendering

Nearest-center culling is not mathematically safe for anisotropic splats. A long tilted ellipsoid may cover a pixel even when many narrow Gaussians have closer centers. The evidence renderer truncates at five standard deviations, builds an exact world-axis bounding box for each support ellipsoid, assigns every primitive to one multilevel tile, queries twenty-seven neighboring cells, and applies the true Mahalanobis test. Beyond five sigma, boundary weight is approximately three-point-seven times ten to the minus six.

**On screen:** Five-sigma tiled support is complete; Euclidean center KNN is not.
