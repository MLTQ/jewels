"""Tests for frozen text-factor resolution."""

from __future__ import annotations

import unittest

import torch

from sol.audit_additive_prompt_jewel_caster import resolve_factor


class AdditivePromptAuditTests(unittest.TestCase):
    def test_resolver_selects_nearest_normalized_factor(self) -> None:
        candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assertEqual(resolve_factor(torch.tensor([[0.1, 2.0]]), candidates), 1)


if __name__ == "__main__":
    unittest.main()
