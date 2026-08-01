"""Primitive-set featurization: fit checkpoints <-> training tensors.

The prior must never see (scale, quat) raw: quaternions double-cover rotations
(q and -q are the same primitive) and eigen-axes permute, so identical
geometry would map to distant feature vectors and the model would spend
capacity learning gauge. Geometry travels instead as the log-covariance
(log-Euclidean SPD coordinates):

    logSigma = R diag(2 log s) R^T      symmetric, unique, 6 numbers

Decode recovers (R, s) by eigendecomposition; ANY valid factorization renders
identically, so the eigh's own sign/permutation arbitrariness is harmless.

Feature layout, FEAT_DIM = 22 per primitive:
    [0:3]   mu (u, v, t)
    [3:9]   upper triangle of logSigma (uu, uv, ut, vv, vt, tt)
    [9:12]  color (P0)
    [12:21] color_grad flattened row-major (P1; world-frame, gauge-free)
    [21]    logit_w
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.params import PrimitiveField, quat_to_rotmat

FEAT_DIM = 22
_IU = torch.triu_indices(3, 3)


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) proper rotations -> (N, 4) unit quaternions (w, x, y, z).

    Batched Shepperd: build all four candidate constructions, keep the one
    with the largest pivot per row (numerically safest branch).
    """
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    piv = 0.5 * torch.stack([
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ], -1).clamp_min(1e-12).sqrt()

    p0, p1, p2, p3 = piv.unbind(-1)
    c0 = torch.stack([p0, (m21 - m12) / (4 * p0), (m02 - m20) / (4 * p0),
                      (m10 - m01) / (4 * p0)], -1)
    c1 = torch.stack([(m21 - m12) / (4 * p1), p1, (m01 + m10) / (4 * p1),
                      (m02 + m20) / (4 * p1)], -1)
    c2 = torch.stack([(m02 - m20) / (4 * p2), (m01 + m10) / (4 * p2), p2,
                      (m12 + m21) / (4 * p2)], -1)
    c3 = torch.stack([(m10 - m01) / (4 * p3), (m02 + m20) / (4 * p3),
                      (m12 + m21) / (4 * p3), p3], -1)

    cands = torch.stack([c0, c1, c2, c3], 1)  # (N, 4, 4)
    idx = piv.argmax(-1)
    q = cands[torch.arange(len(idx)), idx]
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def state_to_features(state: dict) -> torch.Tensor:
    """PrimitiveField state_dict -> (N, 22) float32."""
    mu = state["mu"].float()
    n = mu.shape[0]
    log_scale = state["log_scale"].float().clamp(-8.0, 2.0)
    R = quat_to_rotmat(state["quat"].float())
    lam = 2.0 * log_scale
    log_sigma = torch.einsum("nij,nj,nkj->nik", R, lam, R)
    triu = log_sigma[:, _IU[0], _IU[1]]
    return torch.cat([
        mu, triu, state["color"].float(),
        state["color_grad"].float().reshape(n, 9),
        state["logit_w"].float()[:, None],
    ], 1)


def features_to_field(
    feats: torch.Tensor, *, device: str | torch.device = "cpu"
) -> PrimitiveField:
    """(N, 22) -> renderable PrimitiveField."""
    n = feats.shape[0]
    feats = feats.float().cpu()

    log_sigma = feats.new_zeros(n, 3, 3)
    log_sigma[:, _IU[0], _IU[1]] = feats[:, 3:9]
    log_sigma = log_sigma + log_sigma.transpose(1, 2)
    log_sigma -= torch.diag_embed(
        torch.diagonal(log_sigma, dim1=1, dim2=2) / 2
    )

    lam, U = torch.linalg.eigh(log_sigma)
    det = torch.linalg.det(U)
    U = U.clone()
    U[:, :, 0] = torch.where(det[:, None] < 0, -U[:, :, 0], U[:, :, 0])

    field = PrimitiveField(n, p1_color=True, device="cpu")
    with torch.no_grad():
        field.mu.copy_(feats[:, :3])
        field.log_scale.copy_((lam / 2).clamp(-8.0, 2.0))
        field.quat.copy_(rotmat_to_quat(U))
        field.color.copy_(feats[:, 9:12])
        field.color_grad.copy_(feats[:, 12:21].reshape(n, 3, 3))
        field.logit_w.copy_(feats[:, 21])
    return field.to(device)


def load_corpus(corpus_dir: str | Path) -> dict:
    """Load every checkpoint + CLIP sidecar -> stacked tensors.

    Returns feats (S, N, 22), clip (S, D), bg (S, 3), shape (T, H, W), names.
    Asserts a uniform primitive count — v0 trains on fixed-size sets; padding
    and masks come with the first corpus that needs them.
    """
    corpus_dir = Path(corpus_dir)
    ckpts = sorted(corpus_dir.glob("*_w*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {corpus_dir}")

    feats, clips, bgs, names, shape = [], [], [], [], None
    for ck in ckpts:
        data = torch.load(ck, map_location="cpu", weights_only=False)
        feats.append(state_to_features(data["state"]))
        clips.append(torch.from_numpy(np.load(ck.with_suffix(".clip.npy"))))
        bgs.append(torch.tensor(data["info"]["background"]))
        shape = tuple(data["info"]["shape"])
        names.append(ck.name)

    ns = {f.shape[0] for f in feats}
    if len(ns) != 1:
        raise ValueError(f"non-uniform set sizes {sorted(ns)}; v0 needs fixed N")

    return {
        "feats": torch.stack(feats),
        "clip": torch.stack(clips).float(),
        "bg": torch.stack(bgs).float(),
        "shape": shape,
        "names": names,
    }
