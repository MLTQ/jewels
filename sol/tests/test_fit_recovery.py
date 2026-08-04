"""Regression tests for exact, atomic stage-1 fit recovery."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


STPRIM_ROOT = Path(__file__).resolve().parents[2] / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from fit.fitter import FitConfig, fit_volume  # noqa: E402
from fit.recovery import RECOVERY_SCHEMA, atomic_torch_save  # noqa: E402


class PlannedInterruption(RuntimeError):
    """Simulate a process stopping immediately after a durable checkpoint."""


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append(f"cuda:{torch.cuda.device_count() - 1}")


class FitRecoveryTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_across_densification(self) -> None:
        for device in DEVICES:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as tempdir:
                self._assert_exact_resume(Path(tempdir), device)

    def _assert_exact_resume(self, tempdir: Path, device: str) -> None:
        video = torch.arange(2 * 4 * 4 * 3, dtype=torch.float32).reshape(
            2, 4, 4, 3
        )
        video = video / video.max()
        cfg = FitConfig(
            num_init=8,
            max_primitives=12,
            steps=7,
            lr=0.01,
            voxels_per_step=32,
            knn=4,
            p1_color=True,
            seed=17,
            adapt_every=2,
            adapt_until_frac=1.0,
            densify_frac=0.25,
            log_every=1,
        )

        expected_field, expected_info = fit_volume(
            video, cfg, device=device, verbose=False
        )
        recovery_path = tempdir / "window.recovery.pt"

        def interrupt_after_save(state: dict) -> None:
            atomic_torch_save(
                {"schema": RECOVERY_SCHEMA, "fit_state": state}, recovery_path
            )
            raise PlannedInterruption

        with self.assertRaises(PlannedInterruption):
            fit_volume(
                video,
                cfg,
                device=device,
                verbose=False,
                checkpoint_every=4,
                checkpoint_callback=interrupt_after_save,
            )

        payload = torch.load(recovery_path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["schema"], RECOVERY_SCHEMA)
        state = payload["fit_state"]
        self.assertEqual(state["next_step"], 4)
        self.assertEqual(state["field_state"]["mu"].shape[0], 10)
        self.assertTrue(state["optimizer_state"]["state"])
        self.assertEqual(state["tracker_count"], 1)

        actual_field, actual_info = fit_volume(
            video,
            cfg,
            device=device,
            verbose=False,
            resume_state=state,
        )

        for name, expected in expected_field.state_dict().items():
            self.assertTrue(
                torch.equal(actual_field.state_dict()[name], expected), name
            )
        self.assertEqual(actual_info["history"], expected_info["history"])
        self.assertEqual(actual_info["n_final"], expected_info["n_final"])
        self.assertEqual(actual_info["shape"], expected_info["shape"])
        self.assertEqual(actual_info["background"], expected_info["background"])


if __name__ == "__main__":
    unittest.main()
