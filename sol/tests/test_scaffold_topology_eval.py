"""Tests for topology metrics, calibration, and scaffold controls."""

from __future__ import annotations

import unittest

import torch

from sol.scaffold_topology import ScaffoldTopologyModel, ScaffoldTopologyOutput
from sol.scaffold_topology_eval import (
    TopologyControlView,
    calibrate_occupancy_threshold,
    evaluate_topology_controls,
    expand_topology_counts,
    topology_metrics,
)
from sol.token_grid import GridSpec


class ScaffoldTopologyEvalTests(unittest.TestCase):
    def test_expand_counts_uses_canonical_nested_ranks(self) -> None:
        cells, ranks = expand_topology_counts(
            torch.tensor([2, 0, 3, 1]), slots_per_cell=3
        )
        self.assertTrue(torch.equal(cells, torch.tensor([0, 0, 2, 2, 2, 3])))
        self.assertTrue(torch.equal(ranks, torch.tensor([0, 1, 0, 1, 2, 0])))
        empty_cells, empty_ranks = expand_topology_counts(
            torch.zeros(4, dtype=torch.long), slots_per_cell=3
        )
        self.assertEqual(len(empty_cells), 0)
        self.assertEqual(len(empty_ranks), 0)

    def test_expand_counts_rejects_capacity_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity"):
            expand_topology_counts(torch.tensor([4]), slots_per_cell=3)

    def test_exact_counts_score_one(self) -> None:
        counts = [torch.tensor([2, 0, 3, 1]), torch.tensor([0, 1, 1, 2])]
        metrics = topology_metrics(counts, counts)
        self.assertEqual(metrics["cell_count_mae"], 0.0)
        self.assertAlmostEqual(metrics["total_count_ratio"], 1.0)
        self.assertAlmostEqual(metrics["occupancy_f1"], 1.0)
        self.assertAlmostEqual(metrics["slot_f1"], 1.0)

    def test_training_threshold_calibration_prefers_slot_overlap(self) -> None:
        outputs = [
            ScaffoldTopologyOutput(
                occupancy_logits=torch.tensor([3.0, -3.0, 0.2, -0.2]),
                positive_count_raw=torch.tensor([0.5, 0.5, 0.5, 0.5]),
            )
        ]
        threshold, metrics = calibrate_occupancy_threshold(
            outputs, [torch.tensor([2, 0, 2, 0])], slots_per_cell=8
        )
        self.assertTrue(0 < threshold < 1)
        self.assertGreater(metrics["slot_f1"], 0.5)

    def test_controls_preserve_target_and_rotate_classes(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        model = ScaffoldTopologyModel(
            model_dim=32,
            grid_spec=spec,
            encoder_depth=1,
            cell_depth=1,
        )
        views = []
        for class_id in range(2):
            views.append(
                TopologyControlView(
                    source_id=f"source-{class_id}",
                    class_id=class_id,
                    class_name=f"class-{class_id}",
                    index=0,
                    guide_raster=torch.full(
                        (spec.n_cells, 3), float(class_id + 1) / 3
                    ),
                    carry_raster=torch.zeros(spec.n_cells, 3),
                    target_counts=torch.ones(spec.n_cells, dtype=torch.long),
                )
            )
        report = evaluate_topology_controls(
            model,
            views,
            {0: torch.ones(spec.n_cells)},
            occupancy_threshold=0.5,
            device="cpu",
        )
        self.assertEqual(report["validation_views"], 2)
        self.assertEqual(
            set(report["aggregate"]),
            {"correct", "shuffled", "null", "correct_no_carry", "train_mean"},
        )
        self.assertAlmostEqual(report["aggregate"]["train_mean"]["slot_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
