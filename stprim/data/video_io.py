"""Load a video clip into a (T, H, W, 3) float tensor in [0, 1].

Deliberately dependency-tolerant: tries PyAV, then OpenCV, then imageio. On a
training box one of these is always already installed, and none of them is
worth making a hard requirement.

Scoping guidance for early experiments (see PROJECT.md): fixed camera, no cuts.
Global camera motion shears every primitive simultaneously and will consume the
entire budget before anything is learned about objects; a cut violates the
volume-continuity assumption outright and shows up as a wall of primitives at
the boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _decode_pyav(path: str, max_frames: int | None):
    import av  # noqa

    frames = []
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def _decode_cv2(path: str, max_frames: int | None):
    import cv2  # noqa

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _decode_imageio(path: str, max_frames: int | None):
    import imageio.v3 as iio  # noqa

    frames = []
    for i, f in enumerate(iio.imiter(path)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(f[..., :3])
    return np.stack(frames)


def load_video(
    path: str | Path,
    *,
    max_frames: int | None = 32,
    resize: int | tuple[int, int] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Decode to (T, H, W, 3) float32 in [0, 1].

    `resize` is either an int — target short side, aspect preserved (the source
    aspect isn't known until decode, so it can't be the caller's job) — or an
    explicit (H, W). Downscaling here rather than after loading keeps peak host
    memory bounded for long clips.
    """
    path = str(path)
    errors = []
    for fn in (_decode_pyav, _decode_cv2, _decode_imageio):
        try:
            arr = fn(path, max_frames)
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {e}")
    else:
        raise RuntimeError("no video decoder available:\n  " + "\n  ".join(errors))

    vid = torch.from_numpy(arr).float().div_(255.0)  # (T, H, W, 3)

    if resize is not None:
        if isinstance(resize, int):
            h0, w0 = vid.shape[1], vid.shape[2]
            if h0 <= w0:
                resize = (resize, max(1, round(w0 * resize / h0)))
            else:
                resize = (max(1, round(h0 * resize / w0)), resize)
        vid = torch.nn.functional.interpolate(
            vid.permute(0, 3, 1, 2), size=resize, mode="area"
        ).permute(0, 2, 3, 1).contiguous()

    return vid.to(device)


def synthetic_tube(
    T: int = 24, H: int = 96, W: int = 96, *, device="cpu"
) -> torch.Tensor:
    """A blob translating linearly — i.e. a literal sheared tube in (u,v,t).

    The falsification test for this entire representation. If a handful of
    anisotropic primitives cannot capture this near-perfectly, the premise that
    motion lives in primitive orientation is wrong, and you learn that in
    seconds rather than after a training run.
    """
    t = torch.linspace(-1, 1, T, device=device)[:, None, None]
    v = torch.linspace(-1, 1, H, device=device)[None, :, None]
    u = torch.linspace(-1, 1, W, device=device)[None, None, :]
    blob = torch.exp(-(((u - 0.6 * t) ** 2 + v**2) / 0.08))
    return torch.stack([blob, blob * 0.4, 1 - blob], -1).clamp(0, 1)
