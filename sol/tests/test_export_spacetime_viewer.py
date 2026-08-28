from __future__ import annotations

import torch

from sol.export_spacetime_viewer import build_viewer_payload


def synthetic_checkpoint() -> dict:
    return {
        "state": {
            "mu": torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]]),
            "log_scale": torch.log(torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])),
            "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
            "color": torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.5, 0.3]]),
            "color_grad": torch.zeros(2, 3, 3),
            "logit_w": torch.tensor([0.0, 1.0]),
        },
        "cfg": {"mode": "additive", "steps": 20, "t_scale": 1.25},
        "info": {"shape": (8, 12, 16), "background": [0.1, 0.2, 0.3], "seconds": 2.5},
    }


def test_payload_has_stable_schema_and_strides() -> None:
    payload = build_viewer_payload(
        synthetic_checkpoint(), source_name="field.pt", source_sha256="abc123"
    )

    assert payload["schema"] == "spacetime-jewel-viewer-v1"
    assert payload["field"]["count"] == 2
    assert payload["field"]["shape"] == [8, 12, 16]
    assert payload["source"]["sha256"] == "abc123"
    assert len(payload["arrays"]["centers"]) == 6
    assert len(payload["arrays"]["quaternions"]) == 8
    assert len(payload["arrays"]["color_gradients"]) == 18
    assert len(payload["arrays"]["slice_roots"]) == 8


def test_axis_aligned_slice_derivatives_match_geometry() -> None:
    payload = build_viewer_payload(
        synthetic_checkpoint(), source_name="field.pt", source_sha256="abc123"
    )

    velocity = torch.tensor(payload["arrays"]["slice_velocities"]).reshape(2, 2)
    roots = torch.tensor(payload["arrays"]["slice_roots"]).reshape(2, 2, 2)
    sigmas = torch.tensor(payload["arrays"]["time_sigmas"])
    quaternions = torch.tensor(payload["arrays"]["quaternions"]).reshape(2, 4)

    assert torch.allclose(velocity, torch.zeros_like(velocity), atol=1e-6)
    assert torch.allclose(
        roots[0] @ roots[0].T,
        torch.diag(torch.tensor([0.04, 0.09])),
        atol=1e-6,
    )
    assert torch.allclose(sigmas, torch.tensor([0.4, 0.7]), atol=1e-6)
    assert torch.allclose(quaternions[:, 3], torch.ones(2), atol=1e-6)
