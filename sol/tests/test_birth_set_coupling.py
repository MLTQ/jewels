"""Learned neighborhood birth-set coupling tests."""

from __future__ import annotations

import unittest

import torch

from sol.birth_set_coupling import NeighborhoodBirthSetBlock, rasterize_set_moments


class BirthSetCouplingTests(unittest.TestCase):
    def test_set_moments_retain_count_mean_and_variance(self) -> None:
        values = torch.tensor([[1.0, 3.0], [3.0, 7.0], [9.0, 11.0]])
        cells = torch.tensor([0, 0, 2])
        raster = rasterize_set_moments(values, cells, 4)
        self.assertEqual(raster.shape, (4, 6))
        self.assertTrue(torch.equal(raster[0, :2], torch.tensor([2.0, 5.0])))
        self.assertTrue(torch.equal(raster[0, 2:4], torch.tensor([1.0, 4.0])))
        self.assertAlmostEqual(
            float(raster[0, 4]), torch.log(torch.tensor(3.0)).item()
        )
        self.assertEqual(float(raster[1, 5]), 0.0)

    def test_zero_residual_is_exact_and_receives_gradient(self) -> None:
        block = NeighborhoodBirthSetBlock(32, (2, 2, 2))
        hidden = torch.randn(5, 32, requires_grad=True)
        cells = torch.tensor([0, 0, 1, 4, 7])
        output = block(hidden, cells)
        self.assertTrue(torch.equal(output, hidden))
        output.square().mean().backward()
        self.assertGreater(float(block.row_update[-1].weight.grad.abs().sum()), 0)

    def test_coupling_is_permutation_equivariant(self) -> None:
        torch.manual_seed(4)
        block = NeighborhoodBirthSetBlock(32, (2, 2, 2))
        torch.nn.init.normal_(block.row_update[-1].weight, std=0.01)
        hidden = torch.randn(6, 32)
        cells = torch.tensor([0, 0, 1, 2, 2, 7])
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        direct = block(hidden, cells)
        permuted = block(hidden[permutation], cells[permutation])
        self.assertTrue(torch.allclose(permuted, direct[permutation], atol=1e-6))

    def test_neighboring_cell_state_changes_a_row_update(self) -> None:
        torch.manual_seed(5)
        block = NeighborhoodBirthSetBlock(32, (2, 2, 2))
        torch.nn.init.normal_(block.row_update[-1].weight, std=0.01)
        center = torch.randn(1, 32)
        neighbor_a = torch.zeros(1, 32)
        neighbor_b = torch.ones(1, 32)
        cells = torch.tensor([0, 1])
        output_a = block(torch.cat((center, neighbor_a)), cells)[0]
        output_b = block(torch.cat((center, neighbor_b)), cells)[0]
        self.assertGreater(float((output_a - output_b).abs().max().detach()), 1e-6)


if __name__ == "__main__":
    unittest.main()
