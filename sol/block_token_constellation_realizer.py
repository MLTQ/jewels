"""Map each discrete block token to one complete predefined Jewel constellation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.block_token_language import (
    BlockTokenCodebook,
    block_descriptors,
    block_local_coordinates,
)
from sol.prompt_jewel_caster import ACTIVE_FACTORS, encode_active_jewel_tokens
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class ConstellationBlockRealizer:
    """One medoid local point set and role histogram for every block token."""

    local_centers: torch.Tensor
    jewel_tokens: torch.Tensor
    token_offsets: torch.Tensor
    role_log_probabilities: torch.Tensor
    medoid_field_indices: torch.Tensor
    medoid_block_ids: torch.Tensor
    medoid_distances: torch.Tensor
    block_shape: tuple[int, int, int]
    jitter_std: float

    @property
    def block_vocabulary_size(self) -> int:
        return int(len(self.token_offsets) - 1)

    def state_dict(self) -> dict:
        return {
            "local_centers": self.local_centers.cpu(),
            "jewel_tokens": self.jewel_tokens.cpu(),
            "token_offsets": self.token_offsets.cpu(),
            "role_log_probabilities": self.role_log_probabilities.cpu(),
            "medoid_field_indices": self.medoid_field_indices.cpu(),
            "medoid_block_ids": self.medoid_block_ids.cpu(),
            "medoid_distances": self.medoid_distances.cpu(),
            "block_shape": self.block_shape,
            "jitter_std": self.jitter_std,
        }

    def most_frequent_nonempty_token(self, programs: torch.Tensor) -> int:
        lengths = self.token_offsets[1:] - self.token_offsets[:-1]
        counts = torch.bincount(
            programs.flatten(), minlength=self.block_vocabulary_size
        ).to(lengths)
        counts[lengths <= 0] = -1
        if int(counts.max()) < 0:
            raise RuntimeError("constellation vocabulary has no nonempty token")
        return int(counts.argmax())

    def token_nll(
        self,
        program: torch.Tensor,
        centers: torch.Tensor,
        jewel_tokens: torch.Tensor,
    ) -> dict[str, float]:
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if program.shape != (spec.n_cells,):
            raise ValueError("constellation realization requires one token per block")
        if jewel_tokens.shape != (len(centers), len(ACTIVE_FACTORS)):
            raise ValueError("target Jewel token rows must align with centers")
        block_tokens = program[spec.cell_index(centers)]
        by_role = {}
        for role, name in enumerate(ACTIVE_FACTORS):
            by_role[name] = float(-self.role_log_probabilities[
                block_tokens, role, jewel_tokens[:, role]
            ].mean())
        return {
            "token_nll": by_role,
            "token_nll_macro": sum(by_role.values()) / len(by_role),
        }

    @torch.no_grad()
    def sample(
        self,
        program: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        """Cast all addressed medoid constellations, then adjust only the global count."""
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if program.shape != (spec.n_cells,) or count <= 0:
            raise ValueError("constellation sampling requires a complete program and positive count")
        center_parts, token_parts = [], []
        for block_id, block_token in enumerate(program.tolist()):
            start = int(self.token_offsets[block_token])
            end = int(self.token_offsets[block_token + 1])
            if end == start:
                continue
            local = self.local_centers[start:end]
            if self.jitter_std > 0:
                local = local + torch.randn(
                    local.shape, generator=generator, device=local.device
                ) * self.jitter_std
            local = local.clamp(-0.999, 0.999)
            t = block_id % spec.shape[2]
            y = (block_id // spec.shape[2]) % spec.shape[1]
            x = block_id // (spec.shape[1] * spec.shape[2])
            block_coordinate = local.new_tensor([x, y, t])
            shape = local.new_tensor(spec.shape)
            center_parts.append(
                ((block_coordinate + (local + 1) * 0.5) / shape) * 2 - 1
            )
            token_parts.append(self.jewel_tokens[start:end])
        if not center_parts:
            raise ValueError("constellation program contains no Jewel phrases")
        centers = torch.cat(center_parts)
        tokens = torch.cat(token_parts)
        unadjusted = len(centers)
        if unadjusted > count:
            selected = torch.randperm(
                unadjusted, generator=generator, device=centers.device
            )[:count]
            centers, tokens = centers[selected], tokens[selected]
        elif unadjusted < count:
            extra = torch.randint(
                unadjusted, (count - unadjusted,),
                generator=generator, device=centers.device,
            )
            extra_centers = centers[extra]
            if self.jitter_std > 0:
                extra_centers = (extra_centers + torch.randn(
                    extra_centers.shape, generator=generator,
                    device=centers.device,
                ) * (self.jitter_std / max(spec.shape))).clamp(-0.999, 0.999)
            centers = torch.cat([centers, extra_centers])
            tokens = torch.cat([tokens, tokens[extra]])
        return centers, tokens, {
            "unadjusted_jewels": unadjusted,
            "requested_jewels": count,
            "adjustment_fraction": abs(unadjusted - count) / count,
        }


def fit_constellation_block_realizer(
    fields: list[torch.Tensor],
    programs: torch.Tensor,
    *,
    block_codebook: BlockTokenCodebook,
    physical_codebook,
    smoothing: float = 0.1,
    jitter_std: float = 0.005,
) -> tuple[ConstellationBlockRealizer, dict]:
    """Choose the closest complete training-block occurrence for every token prototype."""
    if len(fields) != len(programs) or not fields:
        raise ValueError("constellation fields and programs must align")
    if smoothing <= 0 or jitter_std < 0:
        raise ValueError("constellation smoothing and jitter are invalid")
    spec = GridSpec(block_codebook.block_shape, slots_per_cell=1)
    if programs.shape[1] != spec.n_cells:
        raise ValueError("constellation programs have an incompatible block count")
    descriptor_rows = []
    for field in fields:
        descriptor = block_descriptors(
            field,
            spec=spec,
            intrinsic_mean=block_codebook.intrinsic_mean,
            intrinsic_std=block_codebook.intrinsic_std,
            local_hist_shape=block_codebook.local_hist_shape,
        )
        descriptor_rows.append(
            (descriptor - block_codebook.descriptor_mean.to(descriptor))
            / block_codebook.descriptor_std.to(descriptor)
        )
    descriptors = torch.stack(descriptor_rows)
    prototypes = block_codebook.prototypes.to(descriptors)
    distances = (
        descriptors - prototypes[programs]
    ).square().mean(dim=2)
    vocabulary_size = block_codebook.vocabulary_size
    medoid_fields = torch.full(
        (vocabulary_size,), -1, dtype=torch.long, device=programs.device
    )
    medoid_blocks = torch.full_like(medoid_fields, -1)
    medoid_distances = torch.full(
        (vocabulary_size,), float("inf"), device=programs.device
    )
    for field_index in range(len(fields)):
        for block_id in range(spec.n_cells):
            token = int(programs[field_index, block_id])
            distance = distances[field_index, block_id]
            if distance < medoid_distances[token]:
                medoid_distances[token] = distance
                medoid_fields[token] = field_index
                medoid_blocks[token] = block_id
    if (medoid_fields < 0).any():
        missing = torch.nonzero(medoid_fields < 0).flatten().tolist()
        raise ValueError(f"constellation vocabulary contains unused tokens: {missing[:8]}")

    local_parts, token_parts, lengths = [], [], []
    for token in range(vocabulary_size):
        field_index = int(medoid_fields[token])
        block_id = int(medoid_blocks[token])
        field = fields[field_index]
        cells, local = block_local_coordinates(field[:, :3], spec)
        selected = cells == block_id
        local_parts.append(local[selected])
        token_parts.append(
            encode_active_jewel_tokens(field[selected], physical_codebook)
        )
        lengths.append(int(selected.sum()))
    local_centers = torch.cat(local_parts) if sum(lengths) else fields[0].new_empty(0, 3)
    jewel_tokens = torch.cat(token_parts) if sum(lengths) else torch.empty(
        0, len(ACTIVE_FACTORS), dtype=torch.long, device=fields[0].device
    )
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=fields[0].device)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=fields[0].device), length_tensor.cumsum(0)]
    )
    histogram = fields[0].new_zeros(
        vocabulary_size, len(ACTIVE_FACTORS), physical_codebook.vocabulary_size
    )
    owner = torch.repeat_interleave(
        torch.arange(vocabulary_size, device=fields[0].device), length_tensor
    )
    for role in range(len(ACTIVE_FACTORS)):
        flat = owner * physical_codebook.vocabulary_size + jewel_tokens[:, role]
        histogram[:, role] = torch.bincount(
            flat,
            minlength=vocabulary_size * physical_codebook.vocabulary_size,
        ).reshape(vocabulary_size, physical_codebook.vocabulary_size)
    probabilities = (histogram + smoothing) / (
        histogram.sum(dim=2, keepdim=True)
        + smoothing * physical_codebook.vocabulary_size
    )
    realizer = ConstellationBlockRealizer(
        local_centers=local_centers,
        jewel_tokens=jewel_tokens,
        token_offsets=offsets,
        role_log_probabilities=probabilities.log(),
        medoid_field_indices=medoid_fields,
        medoid_block_ids=medoid_blocks,
        medoid_distances=medoid_distances,
        block_shape=spec.shape,
        jitter_std=jitter_std,
    )
    return realizer, {
        "training_fields": len(fields),
        "vocabulary_size": vocabulary_size,
        "stored_constellation_jewels": int(len(local_centers)),
        "empty_constellations": int((length_tensor == 0).sum()),
        "mean_constellation_jewels": float(length_tensor.float().mean()),
        "mean_medoid_distance": float(medoid_distances.mean()),
        "smoothing": smoothing,
        "jitter_std": jitter_std,
    }
