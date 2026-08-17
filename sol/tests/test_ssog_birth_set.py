"""Tests for the steered separable Gaussian birth-set coupling."""

from __future__ import annotations

import unittest

import torch

from sol.birth_mark_flow import BirthMarkFlowModel
from sol.birth_set_coupling import SsogBirthSetBlock, rasterize_set_moments
from sol.calibrate_coupled_set_checkpoint import calibrated_checkpoint
from sol.token_grid import GridSpec

GRID = (4, 4, 2)


def _hidden(count: int = 10, dim: int = 16):
    torch.manual_seed(0)
    n_cells = GRID[0] * GRID[1] * GRID[2]
    return torch.randn(count, dim), torch.randint(0, n_cells, (count,))


class SsogBirthSetTests(unittest.TestCase):
    def test_block_is_identity_at_initialization(self) -> None:
        block = SsogBirthSetBlock(16, GRID)
        hidden, cells = _hidden()
        self.assertTrue(torch.equal(block(hidden, cells), hidden))

    def test_separable_field_matches_dense_mixture(self) -> None:
        torch.manual_seed(1)
        block = SsogBirthSetBlock(16, GRID, atoms=2)
        with torch.no_grad():
            block.steer.weight.normal_(0, 0.5)
            block.steer.bias.normal_(0, 0.5)
            block.gate_mu.fill_(0.7)
            block.gate_sigma.fill_(0.4)
            block.gate_lambda.fill_(0.6)
        hidden, cells = _hidden()
        moments = rasterize_set_moments(hidden, cells, block.n_cells)
        context = block._field_context(moments)

        state = block.moment_projection(moments.float())
        raw = block.steer(state).reshape(block.n_cells, block.atoms, 7)
        mu = block.mu0[None] + block.gate_mu * block.max_offset * torch.tanh(
            raw[..., 0:3]
        )
        sigma = torch.exp(
            block.log_sigma0[None] + block.gate_sigma * torch.tanh(raw[..., 3:6])
        ).clamp(0.3, float(max(GRID)))
        weights = torch.softmax(
            block.log_lambda0[None] + block.gate_lambda * torch.tanh(raw[..., 6]),
            dim=-1,
        )
        occupancy = moments[:, -1].float()
        expected = torch.zeros_like(state)
        for destination in range(block.n_cells):
            for atom in range(block.atoms):
                displacement = (
                    block.cell_coordinates
                    - block.cell_coordinates[destination]
                    - mu[destination, atom]
                )
                dense = occupancy * torch.exp(
                    -0.5 * (displacement / sigma[destination, atom]).square().sum(-1)
                )
                expected[destination] += weights[destination, atom] * (
                    dense[None] @ state
                )[0] / dense.sum().clamp_min(1e-6)
        self.assertTrue(torch.allclose(context, expected, atol=1e-4))

    def test_steering_stays_bounded(self) -> None:
        block = SsogBirthSetBlock(16, GRID, atoms=3, max_offset=2.0)
        with torch.no_grad():
            block.steer.weight.normal_(0, 10.0)
            block.gate_mu.fill_(5.0)
        hidden, cells = _hidden()
        moments = rasterize_set_moments(hidden, cells, block.n_cells)
        state = block.moment_projection(moments.float())
        raw = block.steer(state).reshape(block.n_cells, block.atoms, 7)
        mu = block.mu0[None] + block.gate_mu * block.max_offset * torch.tanh(
            raw[..., 0:3]
        )
        travel = (mu - block.mu0[None]).abs().max()
        self.assertLessEqual(float(travel), 5.0 * 2.0 + 1e-5)

    def test_augmented_model_matches_base_and_calibrates(self) -> None:
        spec = GridSpec(GRID, 8)
        common = dict(
            model_dim=16,
            context_depth=1,
            noisy_depth=1,
            guide_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=8,
            guide_dim=3,
        )
        torch.manual_seed(2)
        base = BirthMarkFlowModel(grid_spec=spec, **common)
        torch.manual_seed(2)
        augmented = BirthMarkFlowModel(
            grid_spec=spec, **common, set_depth=1, set_coupling="ssog"
        )
        augmented.load_state_dict(base.state_dict(), strict=False)
        inputs = (
            torch.randn(spec.n_cells, 46),
            torch.randn(6, 22),
            torch.rand(1),
            torch.randint(0, spec.n_cells, (6,)),
            torch.randint(0, spec.slots_per_cell, (6,)),
            torch.randn(8),
        )
        base_out = base(*inputs, guide_raster=torch.rand(spec.n_cells, 3))
        torch.manual_seed(3)
        augmented_out = augmented(
            *inputs, guide_raster=torch.rand(spec.n_cells, 3)
        )
        torch.manual_seed(3)
        base_again = base(*inputs, guide_raster=torch.rand(spec.n_cells, 3))
        self.assertTrue(torch.equal(augmented_out, base_again))
        saved = {
            "model": augmented.state_dict(),
            "meta": {
                "architecture": "scaffold_birth_mark_flow_v1",
                "model_args": {"set_depth": 1, "set_coupling": "ssog"},
            },
        }
        calibrated = calibrated_checkpoint(saved, 0.5, "test")
        self.assertIsNotNone(calibrated)


if __name__ == "__main__":
    unittest.main()
