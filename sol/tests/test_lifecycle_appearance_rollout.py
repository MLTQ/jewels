"""Tests for base-owned topology and lifecycle during autonomous rollout."""

from __future__ import annotations

import copy
import unittest

import torch

from sol.birth_mark_flow import BirthMarkFlowModel
from sol.lifecycle_appearance_flow import (
    APPEARANCE_DIMENSION_SETS,
    LIFECYCLE_DIMENSIONS,
)
from sol.lifecycle_appearance_rollout import rollout_lifecycle_appearance_marks
from sol.scaffold_mark_rollout import rollout_scaffold_marks
from sol.scaffold_topology import ScaffoldTopologyModel
from sol.streaming_data import FeatureStandardizer
from sol.token_grid import GridSpec


class LifecycleAppearanceRolloutTests(unittest.TestCase):
    def test_base_matches_standalone_and_owns_all_temporal_state(self) -> None:
        torch.manual_seed(6)
        spec = GridSpec((2, 2, 2), 4)
        topology = ScaffoldTopologyModel(
            model_dim=8,
            grid_spec=spec,
            encoder_depth=1,
            cell_depth=1,
        )
        with torch.no_grad():
            for parameter in topology.parameters():
                parameter.zero_()
            topology.occupancy_head.bias.fill_(10)
            topology.positive_count_head.bias.fill_(-10)
        base = BirthMarkFlowModel(
            model_dim=8,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            guide_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=8,
            guide_dim=3,
            guide_heads=1,
        )
        appearance = copy.deepcopy(base)
        with torch.no_grad():
            appearance.velocity_head.bias.fill_(0.75)
        standardizer = FeatureStandardizer(torch.zeros(22), torch.ones(22))
        guides = [torch.rand(spec.n_cells, 3) for _ in range(3)]
        text = torch.randn(8)
        common = dict(
            total_frames=24,
            stride_frames=8,
            support_sigma=2.0,
            topology_spec=spec,
            occupancy_threshold=0.5,
            device="cpu",
            steps=2,
        )
        standalone = rollout_scaffold_marks(
            topology,
            base,
            guides,
            text,
            standardizer,
            standardizer,
            generator=torch.Generator().manual_seed(13),
            **common,
        )
        paired = rollout_lifecycle_appearance_marks(
            topology,
            base,
            appearance,
            guides,
            text,
            standardizer,
            standardizer,
            generator=torch.Generator().manual_seed(13),
            appearance_dimensions=APPEARANCE_DIMENSION_SETS["static-detail"],
            appearance_cell_weights=tuple(
                torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0])
                for _ in guides
            ),
            **common,
        )
        self.assertTrue(torch.equal(paired.base.features, standalone.features))
        self.assertTrue(paired.lifecycle_exact)
        self.assertTrue(paired.stable_ids_exact)
        self.assertTrue(paired.topology_exact)
        self.assertTrue(
            torch.equal(
                paired.base.features[:, LIFECYCLE_DIMENSIONS],
                paired.appearance.features[:, LIFECYCLE_DIMENSIONS],
            )
        )
        frozen = tuple(
            index
            for index in range(22)
            if index not in APPEARANCE_DIMENSION_SETS["static-detail"]
        )
        self.assertTrue(
            torch.equal(
                paired.base.features[:, frozen],
                paired.appearance.features[:, frozen],
            )
        )
        self.assertGreater(
            float((paired.base.features - paired.appearance.features).abs().max()),
            0.0,
        )
        self.assertTrue(all(window.lifecycle_exact for window in paired.windows))
        self.assertEqual(paired.report["base"]["total_jewels"], 24)
        self.assertTrue(
            all(0 < window.active_appearance_fraction < 1 for window in paired.windows)
        )


if __name__ == "__main__":
    unittest.main()
