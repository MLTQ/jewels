"""Correctness tests for reference and conservative jewel rendering."""

from __future__ import annotations

import unittest

import torch

from sol.render import (
    audit_truncation,
    covariance_terms,
    render_euclidean_knn,
    render_exact,
    render_truncated,
)
from sol.synthetic import elongated_knn_counterexample, random_jewels


class RenderTests(unittest.TestCase):
    def test_chunked_covariance_matches_single_batch(self) -> None:
        features = random_jewels(17, seed=21)
        expected = covariance_terms(features, eigen_chunk=64)
        chunked = covariance_terms(features, eigen_chunk=3)
        torch.testing.assert_close(chunked[0], expected[0])
        torch.testing.assert_close(chunked[1], expected[1])

    def test_conservative_support_keeps_elongated_contributor(self) -> None:
        features, point = elongated_knn_counterexample()
        exact = render_exact(features, point)
        knn = render_euclidean_knn(features, point, k=64)
        conservative = render_truncated(features, point, support_sigma=5.0)
        self.assertGreater(float((exact - knn).abs().max()), 0.8)
        torch.testing.assert_close(conservative, exact, atol=1e-5, rtol=1e-5)

    def test_five_sigma_truncation_is_auditable(self) -> None:
        features = random_jewels(32, seed=9, scale=0.1)
        points = torch.rand(24, 3) * 2 - 1
        report = audit_truncation(features, points, support_sigma=5.0)
        self.assertLess(report.max_abs_error, 1e-4)


if __name__ == "__main__":
    unittest.main()
