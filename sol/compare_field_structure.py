"""Compare the structure of fitted jewel fields against encoder-produced fields.

The fitter densifies on content and produces sheared tubes (the project's founding
premise); the lattice encoder may produce uniform isotropic blobs that merely
sample the video. This quantifies the difference so a structural encoder has a
concrete target rather than an intuition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sol.amortized_encoder import VideoToJewelEncoder
from sol.render import covariance_terms
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


def structure_report(
    features: torch.Tensor, *, sample: int = 20000, eigen_chunk: int = 4096
) -> dict:
    """Summarize anisotropy, temporal tilt, size, and spatial clustering."""
    if features.ndim != 2 or features.shape[1] < 22:
        raise ValueError("features must have shape (N,22+)")
    if sample <= 0 or eigen_chunk <= 0:
        raise ValueError("structure sample and eigen chunk must be positive")
    weight = torch.sigmoid(features[:, 21])
    keep = weight > 0.02
    kept = features[keep]
    if not len(kept):
        raise ValueError("no jewels above the opacity floor")
    if len(kept) > sample:
        index = torch.randperm(len(kept))[:sample]
        kept = kept[index]
    covariance, _ = covariance_terms(kept)
    # CUDA's batched double-precision solver may reserve workspace proportional
    # to the full batch (over 10 GiB for 20k tiny matrices on an RTX 2070).
    # These covariances originate in float32, so solve them in bounded float32
    # chunks and concatenate the same 3x3 eigensystems.
    eigenvalues, eigenvectors = [], []
    for start in range(0, len(covariance), eigen_chunk):
        values, vectors = torch.linalg.eigh(
            covariance[start : start + eigen_chunk].float()
        )
        eigenvalues.append(values)
        eigenvectors.append(vectors)
    eigenvalues = torch.cat(eigenvalues)
    eigenvectors = torch.cat(eigenvectors)
    eigenvalues = eigenvalues.clamp_min(1e-12)
    principal = eigenvectors[:, :, -1]
    anisotropy = (eigenvalues[:, -1] / eigenvalues[:, 0]).sqrt()
    temporal_tilt = principal[:, 2].abs()
    mixed_spacetime_tilt = 2.0 * temporal_tilt * torch.sqrt(
        (1.0 - temporal_tilt.square()).clamp_min(0.0)
    )
    volume = eigenvalues.prod(dim=1).pow(1 / 6)
    centers = kept[:, :3]
    grid = GridSpec((8, 8, 4), 1)
    occupancy = torch.bincount(
        grid.cell_index(centers), minlength=grid.n_cells
    ).float()
    share = occupancy / occupancy.sum().clamp_min(1)
    entropy = float(-(share[share > 0] * share[share > 0].log()).sum())
    return {
        "jewels_total": int(len(features)),
        "jewels_above_2pct_opacity": int(keep.sum()),
        "anisotropy_median": float(anisotropy.median()),
        "anisotropy_p90": float(anisotropy.quantile(0.9)),
        "temporal_tilt_median": float(temporal_tilt.median()),
        "mixed_spacetime_tilt_median": float(mixed_spacetime_tilt.median()),
        "mixed_spacetime_tilt_p90": float(mixed_spacetime_tilt.quantile(0.9)),
        "extent_median": float(volume.median()),
        "extent_iqr_ratio": float(
            volume.quantile(0.75) / volume.quantile(0.25).clamp_min(1e-9)
        ),
        "opacity_median": float(weight[keep].median()),
        "occupancy_entropy": entropy,
        "occupancy_entropy_max": math.log(grid.n_cells),
        "occupancy_uniformity": entropy / math.log(grid.n_cells),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted", action="append", required=True)
    parser.add_argument("--video", action="append", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--frames", type=int, default=49)
    args = parser.parse_args()
    if len(args.fitted) != len(args.video):
        raise ValueError("each fitted field needs its matching video")

    device = torch.device(args.device)
    saved = torch.load(args.encoder, map_location=device, weights_only=False)
    meta = saved["meta"]
    encoder = VideoToJewelEncoder(
        grid_spec=GridSpec(tuple(meta["grid_shape"]), 1024), **meta["model_args"]
    ).to(device)
    encoder.load_state_dict(saved["model"])
    encoder.eval()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stprim"))
    from prior.featurize import state_to_features  # noqa: PLC0415

    records = []
    for fitted_path, video_path in zip(args.fitted, args.video):
        fitted = torch.load(fitted_path, map_location="cpu", weights_only=False)
        fitted_features = state_to_features(fitted["state"]).float()
        video = load_video(
            video_path,
            max_frames=args.frames,
            start_frame=0,
            resize=(args.height, args.width),
            device="cpu",
        ).to(device)
        with torch.no_grad():
            prediction = encoder(video)
            encoder_features = encoder.canonical_features(prediction).cpu()
        records.append(
            {
                "video": video_path,
                "fitted": structure_report(fitted_features),
                "encoder": structure_report(encoder_features),
            }
        )
        print("compared", Path(video_path).stem, flush=True)

    keys = list(records[0]["fitted"])
    macro = {
        arm: {
            k: sum(r[arm][k] for r in records) / len(records)
            for k in keys
            if isinstance(records[0][arm][k], (int, float))
        }
        for arm in ("fitted", "encoder")
    }
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(
            {"schema": "field-structure-comparison-v1", "macro": macro, "records": records},
            indent=1,
        )
    )
    print(f"{'metric':28s} {'fitted':>12s} {'encoder':>12s}")
    for key in keys:
        print(f"{key:28s} {macro['fitted'][key]:12.4f} {macro['encoder'][key]:12.4f}")


if __name__ == "__main__":
    main()
