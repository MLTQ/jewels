"""Tests for canonical field-structure metrics."""

from __future__ import annotations

import math
import unittest

import torch

from sol.compare_field_structure import structure_report


class FieldStructureTests(unittest.TestCase):
    def test_mixed_tilt_detects_diagonal_spacetime_axis(self) -> None:
        axis = torch.tensor([1.0, 0.0, 1.0]) / math.sqrt(2.0)
        middle = torch.tensor([0.0, 1.0, 0.0])
        third = torch.linalg.cross(axis, middle)
        rotation = torch.stack((axis, middle, third), dim=1)
        covariance = rotation @ torch.diag(torch.tensor([4.0, 1.0, 1.0])) @ rotation.T
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
        log_sigma = torch.einsum(
            "ij,j,kj->ik", eigenvectors, eigenvalues.log(), eigenvectors
        ).float()
        packed = torch.tensor([
            log_sigma[0, 0], log_sigma[0, 1], log_sigma[0, 2],
            log_sigma[1, 1], log_sigma[1, 2], log_sigma[2, 2],
        ])
        features = torch.zeros(32, 22)
        features[:, :3] = torch.rand(32, 3) * 2 - 1
        features[:, 3:9] = packed
        features[:, 21] = 0.0
        report = structure_report(features)
        self.assertGreater(report["mixed_spacetime_tilt_median"], 0.99)


if __name__ == "__main__":
    unittest.main()
