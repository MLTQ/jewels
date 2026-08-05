"""Affine time-coordinate transforms for canonical spacetime jewel features."""

from __future__ import annotations

import torch


_UPPER = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _unpack_log_covariance(features: torch.Tensor) -> torch.Tensor:
    matrix = features.new_zeros(features.shape[0], 3, 3)
    for offset, (row, column) in enumerate(_UPPER):
        matrix[:, row, column] = features[:, 3 + offset]
        matrix[:, column, row] = features[:, 3 + offset]
    return matrix


def _pack_log_covariance(matrix: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [matrix[:, row, column] for row, column in _UPPER], dim=1
    )


def _symmetric_exp(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix.double())
    return torch.einsum("nij,nj,nkj->nik", vectors, values.exp(), vectors)


def _symmetric_log(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix.double())
    values = values.clamp_min(1e-16).log()
    return torch.einsum("nij,nj,nkj->nik", vectors, values, vectors)


def _time_affine(total_frames: int, frontier: int, stride_frames: int) -> tuple[float, float]:
    if total_frames < 2 or stride_frames <= 0:
        raise ValueError("total_frames must exceed one and stride_frames must be positive")
    if not 0 <= frontier < total_frames:
        raise ValueError("frontier must lie inside the fitted clip")
    scale = (total_frames - 1) / (2.0 * stride_frames)
    offset = ((total_frames - 1) / 2.0 - frontier) / stride_frames
    return scale, offset


def _transform_time(
    features: torch.Tensor,
    *,
    scale: float,
    offset: float,
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("features must have shape (jewels, 22)")
    if scale <= 0:
        raise ValueError("time scale must be positive")
    if not len(features):
        return features.clone()
    output = features.clone()
    output[:, 2] = features[:, 2] * scale + offset
    transform = torch.diag(features.new_tensor([1.0, 1.0, scale])).double()
    covariance = _symmetric_exp(_unpack_log_covariance(features))
    transformed_covariance = transform @ covariance @ transform.T
    output[:, 3:9] = _pack_log_covariance(
        _symmetric_log(transformed_covariance)
    ).to(features.dtype)
    gradient = features[:, 12:21].reshape(-1, 3, 3)
    inverse = torch.diag(features.new_tensor([1.0, 1.0, 1.0 / scale]))
    output[:, 12:21] = (gradient @ inverse).reshape(-1, 9)
    return output


def to_frontier_time(
    features: torch.Tensor,
    total_frames: int,
    frontier: int,
    stride_frames: int,
) -> torch.Tensor:
    """Map global fitted time into stride units relative to a continuation frontier."""
    scale, offset = _time_affine(total_frames, frontier, stride_frames)
    return _transform_time(features, scale=scale, offset=offset)


def to_global_time(
    features: torch.Tensor,
    total_frames: int,
    frontier: int,
    stride_frames: int,
) -> torch.Tensor:
    """Invert `to_frontier_time` exactly up to symmetric eigensolver precision."""
    scale, offset = _time_affine(total_frames, frontier, stride_frames)
    return _transform_time(features, scale=1.0 / scale, offset=-offset / scale)
