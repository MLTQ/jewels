"""Finite empirical mappings from block tokens to local continuous Jewel phrases."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.block_token_language import block_local_coordinates
from sol.prompt_jewel_caster import ACTIVE_FACTORS, encode_active_jewel_tokens
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class EmpiricalBlockRealizer:
    """CSR reservoirs and count/token statistics for reusable macro-Jewel tokens."""

    phrase_local_centers: torch.Tensor
    phrase_jewel_tokens: torch.Tensor
    phrase_offsets: torch.Tensor
    mean_jewels_per_occurrence: torch.Tensor
    role_log_probabilities: torch.Tensor
    block_shape: tuple[int, int, int]
    jitter_std: float

    @property
    def block_vocabulary_size(self) -> int:
        return int(len(self.mean_jewels_per_occurrence))

    @property
    def jewel_vocabulary_size(self) -> int:
        return int(self.role_log_probabilities.shape[2])

    def state_dict(self) -> dict:
        return {
            "phrase_local_centers": self.phrase_local_centers.cpu(),
            "phrase_jewel_tokens": self.phrase_jewel_tokens.cpu(),
            "phrase_offsets": self.phrase_offsets.cpu(),
            "mean_jewels_per_occurrence": self.mean_jewels_per_occurrence.cpu(),
            "role_log_probabilities": self.role_log_probabilities.cpu(),
            "block_shape": self.block_shape,
            "jitter_std": self.jitter_std,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict, device: torch.device | str = "cpu"
    ) -> "EmpiricalBlockRealizer":
        return cls(
            phrase_local_centers=state["phrase_local_centers"].to(device),
            phrase_jewel_tokens=state["phrase_jewel_tokens"].to(device),
            phrase_offsets=state["phrase_offsets"].to(device),
            mean_jewels_per_occurrence=state["mean_jewels_per_occurrence"].to(device),
            role_log_probabilities=state["role_log_probabilities"].to(device),
            block_shape=tuple(state["block_shape"]),
            jitter_std=float(state["jitter_std"]),
        )

    def most_frequent_nonempty_token(self, programs: torch.Tensor) -> int:
        counts = torch.bincount(
            programs.flatten(), minlength=self.block_vocabulary_size
        ).to(self.mean_jewels_per_occurrence)
        counts[self.mean_jewels_per_occurrence <= 0] = -1
        if float(counts.max()) < 0:
            raise RuntimeError("empirical block vocabulary has no nonempty token")
        return int(counts.argmax())

    def token_nll(
        self,
        program: torch.Tensor,
        centers: torch.Tensor,
        jewel_tokens: torch.Tensor,
    ) -> dict[str, float]:
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if program.shape != (spec.n_cells,):
            raise ValueError("empirical realization requires one token per block")
        if jewel_tokens.shape != (len(centers), len(ACTIVE_FACTORS)):
            raise ValueError("target Jewel token rows must align with centers")
        block_tokens = program[spec.cell_index(centers)]
        role_nll = {}
        for role, name in enumerate(ACTIVE_FACTORS):
            values = self.role_log_probabilities[
                block_tokens, role, jewel_tokens[:, role]
            ]
            role_nll[name] = float(-values.mean())
        return {
            "token_nll": role_nll,
            "token_nll_macro": sum(role_nll.values()) / len(role_nll),
        }

    @torch.no_grad()
    def sample(
        self,
        program: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast complete local Jewel tuples from a finite block-token program."""
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if program.shape != (spec.n_cells,) or count <= 0:
            raise ValueError("empirical sampling requires a complete program and positive count")
        block_weights = self.mean_jewels_per_occurrence[program].clamp_min(0)
        if float(block_weights.sum()) <= 0:
            raise ValueError("block program contains no realizable Jewel mass")
        block_ids = torch.multinomial(
            block_weights, count, replacement=True, generator=generator
        )
        block_tokens = program[block_ids]
        starts = self.phrase_offsets[block_tokens]
        lengths = self.phrase_offsets[block_tokens + 1] - starts
        if (lengths <= 0).any():
            raise RuntimeError("positive-mass block token has an empty phrase reservoir")
        unit = torch.rand(count, device=program.device, generator=generator)
        phrase_rows = starts + torch.floor(unit * lengths).long()
        local = self.phrase_local_centers[phrase_rows]
        if self.jitter_std > 0:
            local = local + torch.randn(
                local.shape, device=local.device, generator=generator
            ) * self.jitter_std
        local = local.clamp(-0.999, 0.999)
        t = block_ids % spec.shape[2]
        y = (block_ids // spec.shape[2]) % spec.shape[1]
        x = block_ids // (spec.shape[1] * spec.shape[2])
        block_coordinates = torch.stack([x, y, t], dim=1).to(local)
        shape = local.new_tensor(spec.shape)
        centers = ((block_coordinates + (local + 1) * 0.5) / shape) * 2 - 1
        return centers, self.phrase_jewel_tokens[phrase_rows]


def fit_empirical_block_realizer(
    fields: list[torch.Tensor],
    programs: torch.Tensor,
    *,
    physical_codebook,
    block_vocabulary_size: int,
    smoothing: float = 0.1,
    jitter_std: float = 0.01,
) -> tuple[EmpiricalBlockRealizer, dict]:
    """Pool target-free-at-use local Jewel tuples by their frozen training block token."""
    if len(fields) != len(programs) or not fields:
        raise ValueError("training fields and block programs must align")
    if smoothing <= 0 or jitter_std < 0 or block_vocabulary_size <= 1:
        raise ValueError("empirical realizer settings are invalid")
    spec = GridSpec(tuple(physical_codebook.grid_shape), slots_per_cell=1)
    if programs.shape[1] != spec.n_cells:
        raise ValueError("training programs have an incompatible block count")
    local_parts, jewel_parts, owner_parts = [], [], []
    for field, program in zip(fields, programs):
        cells, local = block_local_coordinates(field[:, :3], spec)
        local_parts.append(local)
        jewel_parts.append(encode_active_jewel_tokens(field, physical_codebook))
        owner_parts.append(program[cells])
    local = torch.cat(local_parts)
    jewel_tokens = torch.cat(jewel_parts)
    owners = torch.cat(owner_parts)
    order = torch.argsort(owners, stable=True)
    owners = owners[order]
    local = local[order]
    jewel_tokens = jewel_tokens[order]
    phrase_counts = torch.bincount(owners, minlength=block_vocabulary_size)
    phrase_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=owners.device), phrase_counts.cumsum(0)]
    )
    occurrence_counts = torch.bincount(
        programs.flatten(), minlength=block_vocabulary_size
    ).to(local)
    mean_counts = phrase_counts.to(local) / occurrence_counts.clamp_min(1)
    mean_counts[occurrence_counts == 0] = 0

    histogram = local.new_zeros(
        block_vocabulary_size, len(ACTIVE_FACTORS),
        physical_codebook.vocabulary_size,
    )
    for role in range(len(ACTIVE_FACTORS)):
        flat = owners * physical_codebook.vocabulary_size + jewel_tokens[:, role]
        histogram[:, role] = torch.bincount(
            flat,
            minlength=block_vocabulary_size * physical_codebook.vocabulary_size,
        ).reshape(block_vocabulary_size, physical_codebook.vocabulary_size)
    probabilities = (histogram + smoothing) / (
        histogram.sum(dim=2, keepdim=True)
        + smoothing * physical_codebook.vocabulary_size
    )
    realizer = EmpiricalBlockRealizer(
        phrase_local_centers=local,
        phrase_jewel_tokens=jewel_tokens,
        phrase_offsets=phrase_offsets,
        mean_jewels_per_occurrence=mean_counts,
        role_log_probabilities=probabilities.log(),
        block_shape=spec.shape,
        jitter_std=jitter_std,
    )
    utilized = phrase_counts > 0
    return realizer, {
        "training_fields": len(fields),
        "training_phrases": int(len(local)),
        "utilized_tokens": int(utilized.sum()),
        "utilized_fraction": float(utilized.float().mean()),
        "mean_reservoir_phrases_per_utilized_token": float(
            phrase_counts[utilized].float().mean()
        ),
        "smoothing": smoothing,
        "jitter_std": jitter_std,
    }
