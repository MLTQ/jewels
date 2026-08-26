"""Tests for constellation-oracle count adjustment reporting."""

from __future__ import annotations

import unittest

from sol.audit_block_token_constellation_oracle import adjustment_macro


class ConstellationOracleAuditTests(unittest.TestCase):
    def test_adjustment_macro_uses_named_arm(self) -> None:
        rows = [
            {"arms": {"oracle block": {"realization": {
                "unadjusted_jewels": 70000, "adjustment_fraction": 0.02,
            }}}},
            {"arms": {"oracle block": {"realization": {
                "unadjusted_jewels": 74000, "adjustment_fraction": 0.03,
            }}}},
        ]
        report = adjustment_macro(rows)
        self.assertEqual(report["mean_unadjusted_jewels"], 72000)
        self.assertAlmostEqual(report["mean_adjustment_fraction"], 0.025)
        self.assertEqual(report["max_adjustment_fraction"], 0.03)


if __name__ == "__main__":
    unittest.main()
