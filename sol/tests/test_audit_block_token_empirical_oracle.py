"""Tests for empirical macro-Jewel oracle control aggregation."""

from __future__ import annotations

import unittest

import torch

from sol.audit_block_token_empirical_oracle import empirical_control_metrics, realizer_device
from sol.audit_jewel_casting_language import FieldRecord


class _FakePhysical:
    vocabulary_size = 5


class _FakeRealizer:
    def token_nll(self, program, centers, jewel_tokens):
        value = float(program.float().mean())
        roles = {name: value for name in ("covariance", "surface", "gradient")}
        return {"token_nll": roles, "token_nll_macro": value}


class EmpiricalBlockOracleAuditTests(unittest.TestCase):
    def test_device_resolution_covers_both_realizer_schemas(self) -> None:
        empirical = type("Empirical", (), {"phrase_local_centers": torch.zeros(1)})()
        constellation = type("Constellation", (), {"local_centers": torch.zeros(1)})()
        self.assertEqual(realizer_device(empirical), torch.device("cpu"))
        self.assertEqual(realizer_device(constellation), torch.device("cpu"))

    def test_control_program_ownership_is_preserved(self) -> None:
        records = [FieldRecord(
            path="/unused", source_id="s", style="x", fit_seed=1,
            features=torch.zeros(4, 22), background=torch.zeros(3),
        )]
        programs = torch.ones(1, 4, dtype=torch.long)
        shuffled = torch.full((1, 4), 2, dtype=torch.long)
        original = __import__(
            "sol.audit_block_token_empirical_oracle", fromlist=["encode_active_jewel_tokens"]
        ).encode_active_jewel_tokens
        module = __import__("sol.audit_block_token_empirical_oracle", fromlist=["x"])
        module.encode_active_jewel_tokens = lambda features, codebook: torch.zeros(4, 3, dtype=torch.long)
        try:
            report = empirical_control_metrics(
                _FakeRealizer(), records, programs, shuffled, 3,
                physical_codebook=_FakePhysical(), device=torch.device("cpu"),
            )
        finally:
            module.encode_active_jewel_tokens = original
        self.assertEqual(report["oracle block"]["token_nll_macro"], 1.0)
        self.assertEqual(report["shuffled block"]["token_nll_macro"], 2.0)
        self.assertEqual(report["null block"]["token_nll_macro"], 3.0)


if __name__ == "__main__":
    unittest.main()
