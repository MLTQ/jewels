"""Address-conditioned scene/block tokens selecting complete local Jewel constellations."""

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
class AddressedSceneBlockRealizer:
    """Select same-address training constellations by scene and local block token."""

    normalized_descriptors: torch.Tensor
    block_prototypes: torch.Tensor
    block_role_histograms: torch.Tensor
    field_local_centers: torch.Tensor
    field_jewel_tokens: torch.Tensor
    field_block_offsets: torch.Tensor
    scene_owners: torch.Tensor
    block_shape: tuple[int, int, int]
    semantic_scene_tokens: int
    smoothing: float
    jitter_std: float

    @property
    def local_centers(self) -> torch.Tensor:
        """Expose the common storage device used by generic hierarchy audits."""
        return self.field_local_centers

    @property
    def block_vocabulary_size(self) -> int:
        return int(len(self.block_prototypes))

    @property
    def scene_vocabulary_size(self) -> int:
        return self.semantic_scene_tokens + 1

    @property
    def null_scene_token(self) -> int:
        return self.semantic_scene_tokens

    def state_dict(self) -> dict:
        return {
            "normalized_descriptors": self.normalized_descriptors.cpu(),
            "block_prototypes": self.block_prototypes.cpu(),
            "block_role_histograms": self.block_role_histograms.cpu(),
            "field_local_centers": self.field_local_centers.cpu(),
            "field_jewel_tokens": self.field_jewel_tokens.cpu(),
            "field_block_offsets": self.field_block_offsets.cpu(),
            "scene_owners": self.scene_owners.cpu(),
            "block_shape": self.block_shape,
            "semantic_scene_tokens": self.semantic_scene_tokens,
            "smoothing": self.smoothing,
            "jitter_std": self.jitter_std,
        }

    def eligible_fields(self, scene_token: int) -> torch.Tensor:
        if not 0 <= scene_token < self.scene_vocabulary_size:
            raise ValueError("scene token is out of range")
        if scene_token == self.null_scene_token:
            return torch.arange(len(self.scene_owners), device=self.scene_owners.device)
        return torch.nonzero(self.scene_owners == scene_token).flatten()

    def nearest_fields(
        self, scene_token: int, program: torch.Tensor, neighbors: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest eligible field rows for every fixed block address."""
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if program.shape != (spec.n_cells,):
            raise ValueError("addressed realization requires one token per block")
        eligible = self.eligible_fields(scene_token)
        if not 0 < neighbors <= len(eligible):
            raise ValueError("requested neighbor count exceeds eligible scene fields")
        prototypes = self.block_prototypes[program]
        distances = (
            self.normalized_descriptors[eligible] - prototypes[None]
        ).square().mean(dim=2)
        values, rows = torch.topk(distances, neighbors, dim=0, largest=False)
        return eligible[rows], values

    def most_frequent_nonempty_token(
        self, programs: torch.Tensor, scene_token: int | None = None
    ) -> int:
        scene = self.null_scene_token if scene_token is None else scene_token
        counts = torch.bincount(
            programs.flatten(), minlength=self.block_vocabulary_size
        )
        # Same-address selection guarantees at least one candidate; prefer a token used in training.
        return int(counts.argmax())

    def token_nll(
        self,
        scene_token: int,
        program: torch.Tensor,
        centers: torch.Tensor,
        jewel_tokens: torch.Tensor,
    ) -> dict[str, float]:
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if jewel_tokens.shape != (len(centers), len(ACTIVE_FACTORS)):
            raise ValueError("target Jewel tokens must align with centers")
        neighbors = min(4, len(self.eligible_fields(scene_token)))
        fields, _ = self.nearest_fields(scene_token, program, neighbors)
        cells = torch.arange(spec.n_cells, device=program.device)
        histograms = self.block_role_histograms[
            fields, cells[None]
        ].float().sum(dim=0)
        probabilities = (histograms + self.smoothing) / (
            histograms.sum(dim=2, keepdim=True)
            + self.smoothing * histograms.shape[2]
        )
        target_cells = spec.cell_index(centers)
        by_role = {}
        for role, name in enumerate(ACTIVE_FACTORS):
            by_role[name] = float(-probabilities[
                target_cells, role, jewel_tokens[:, role]
            ].clamp_min(1e-12).log().mean())
        return {
            "token_nll": by_role,
            "token_nll_macro": sum(by_role.values()) / len(by_role),
        }

    @torch.no_grad()
    def sample(
        self,
        scene_token: int,
        program: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        """Cast the nearest complete same-address constellation for every program token."""
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if count <= 0:
            raise ValueError("addressed sampling requires a positive Jewel count")
        fields, distances = self.nearest_fields(scene_token, program, 1)
        fields = fields[0]
        centers_out, tokens_out = [], []
        for block_id in range(spec.n_cells):
            field = int(fields[block_id])
            start = int(self.field_block_offsets[field, block_id])
            end = int(self.field_block_offsets[field, block_id + 1])
            if end == start:
                continue
            local = self.field_local_centers[field, start:end]
            if self.jitter_std > 0:
                local = local + torch.randn(
                    local.shape, generator=generator, device=local.device
                ) * self.jitter_std
            local = local.clamp(-0.999, 0.999)
            t = block_id % spec.shape[2]
            y = (block_id // spec.shape[2]) % spec.shape[1]
            x = block_id // (spec.shape[1] * spec.shape[2])
            address = local.new_tensor([x, y, t])
            shape = local.new_tensor(spec.shape)
            centers_out.append(((address + (local + 1) * 0.5) / shape) * 2 - 1)
            tokens_out.append(self.field_jewel_tokens[field, start:end])
        if not centers_out:
            raise ValueError("addressed program contains no Jewel phrases")
        centers = torch.cat(centers_out)
        tokens = torch.cat(tokens_out)
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
                    extra_centers.shape, generator=generator, device=centers.device
                ) * (self.jitter_std / max(spec.shape))).clamp(-0.999, 0.999)
            centers = torch.cat([centers, extra_centers])
            tokens = torch.cat([tokens, tokens[extra]])
        return centers, tokens, {
            "scene_token": scene_token,
            "unadjusted_jewels": unadjusted,
            "requested_jewels": count,
            "adjustment_fraction": abs(unadjusted - count) / count,
            "mean_medoid_distance": float(distances.mean()),
        }


def fit_addressed_scene_block_realizer(
    fields: list[torch.Tensor],
    scene_owners: torch.Tensor,
    *,
    block_codebook: BlockTokenCodebook,
    physical_codebook,
    smoothing: float = 0.1,
    jitter_std: float = 0.005,
) -> tuple[AddressedSceneBlockRealizer, dict]:
    """Prepare same-address descriptors, histograms, and complete training constellations."""
    if scene_owners.shape != (len(fields),) or not fields:
        raise ValueError("addressed fields and scene owners must align")
    semantic_scenes = int(scene_owners.max()) + 1
    if sorted(scene_owners.unique().tolist()) != list(range(semantic_scenes)):
        raise ValueError("semantic scene tokens must be dense from zero")
    if smoothing <= 0 or jitter_std < 0:
        raise ValueError("addressed realization settings are invalid")
    spec = GridSpec(block_codebook.block_shape, slots_per_cell=1)
    descriptors, sorted_locals, sorted_tokens, offsets, histograms = [], [], [], [], []
    expected_jewels = len(fields[0])
    if any(len(field) != expected_jewels for field in fields):
        raise ValueError("addressed realizer currently requires equal field sizes")
    for field in fields:
        descriptor = block_descriptors(
            field,
            spec=spec,
            intrinsic_mean=block_codebook.intrinsic_mean,
            intrinsic_std=block_codebook.intrinsic_std,
            local_hist_shape=block_codebook.local_hist_shape,
        )
        descriptors.append(
            (descriptor - block_codebook.descriptor_mean.to(descriptor))
            / block_codebook.descriptor_std.to(descriptor)
        )
        cells, local = block_local_coordinates(field[:, :3], spec)
        jewel = encode_active_jewel_tokens(field, physical_codebook)
        order = torch.argsort(cells, stable=True)
        counts = torch.bincount(cells, minlength=spec.n_cells)
        offsets.append(torch.cat([
            torch.zeros(1, dtype=torch.long, device=field.device), counts.cumsum(0)
        ]))
        sorted_locals.append(local[order])
        sorted_tokens.append(jewel[order])
        histogram = field.new_zeros(
            spec.n_cells, len(ACTIVE_FACTORS), physical_codebook.vocabulary_size
        )
        for role in range(len(ACTIVE_FACTORS)):
            flat = cells * physical_codebook.vocabulary_size + jewel[:, role]
            histogram[:, role] = torch.bincount(
                flat,
                minlength=spec.n_cells * physical_codebook.vocabulary_size,
            ).reshape(spec.n_cells, physical_codebook.vocabulary_size)
        histograms.append(histogram.half())
    realizer = AddressedSceneBlockRealizer(
        normalized_descriptors=torch.stack(descriptors),
        block_prototypes=block_codebook.prototypes.to(fields[0]),
        block_role_histograms=torch.stack(histograms),
        field_local_centers=torch.stack(sorted_locals),
        field_jewel_tokens=torch.stack(sorted_tokens),
        field_block_offsets=torch.stack(offsets),
        scene_owners=scene_owners,
        block_shape=spec.shape,
        semantic_scene_tokens=semantic_scenes,
        smoothing=smoothing,
        jitter_std=jitter_std,
    )
    return realizer, {
        "training_fields": len(fields),
        "semantic_scene_tokens": semantic_scenes,
        "scene_vocabulary_size_with_null": semantic_scenes + 1,
        "block_vocabulary_size": block_codebook.vocabulary_size,
        "blocks_per_program": spec.n_cells,
        "same_address_candidates_per_semantic_scene": {
            str(scene): int((scene_owners == scene).sum())
            for scene in range(semantic_scenes)
        },
        "likelihood_neighbors": 4,
        "smoothing": smoothing,
        "jitter_std": jitter_std,
    }
