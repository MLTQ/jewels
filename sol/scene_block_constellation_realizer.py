"""Hierarchical scene-token and block-token realization of complete Jewel constellations."""

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
class SceneBlockConstellationRealizer:
    """Predefined complete constellations indexed by a scene/block token pair."""

    local_centers: torch.Tensor
    jewel_tokens: torch.Tensor
    pair_offsets: torch.Tensor
    role_log_probabilities: torch.Tensor
    medoid_field_indices: torch.Tensor
    medoid_block_ids: torch.Tensor
    medoid_distances: torch.Tensor
    block_shape: tuple[int, int, int]
    semantic_scene_tokens: int
    jitter_std: float

    @property
    def scene_vocabulary_size(self) -> int:
        return int(self.role_log_probabilities.shape[0])

    @property
    def block_vocabulary_size(self) -> int:
        return int(self.role_log_probabilities.shape[1])

    @property
    def null_scene_token(self) -> int:
        return self.semantic_scene_tokens

    def state_dict(self) -> dict:
        return {
            "local_centers": self.local_centers.cpu(),
            "jewel_tokens": self.jewel_tokens.cpu(),
            "pair_offsets": self.pair_offsets.cpu(),
            "role_log_probabilities": self.role_log_probabilities.cpu(),
            "medoid_field_indices": self.medoid_field_indices.cpu(),
            "medoid_block_ids": self.medoid_block_ids.cpu(),
            "medoid_distances": self.medoid_distances.cpu(),
            "block_shape": self.block_shape,
            "semantic_scene_tokens": self.semantic_scene_tokens,
            "jitter_std": self.jitter_std,
        }

    def most_frequent_nonempty_token(
        self, programs: torch.Tensor, scene_token: int | None = None
    ) -> int:
        scene = self.null_scene_token if scene_token is None else scene_token
        if not 0 <= scene < self.scene_vocabulary_size:
            raise ValueError("scene token is out of range")
        starts = self.pair_offsets[
            scene * self.block_vocabulary_size:
            (scene + 1) * self.block_vocabulary_size
        ]
        ends = self.pair_offsets[
            scene * self.block_vocabulary_size + 1:
            (scene + 1) * self.block_vocabulary_size + 1
        ]
        counts = torch.bincount(
            programs.flatten(), minlength=self.block_vocabulary_size
        ).to(starts)
        counts[(ends - starts) <= 0] = -1
        if int(counts.max()) < 0:
            raise RuntimeError("scene/block vocabulary has no nonempty token")
        return int(counts.argmax())

    def token_nll(
        self,
        scene_token: int,
        program: torch.Tensor,
        centers: torch.Tensor,
        jewel_tokens: torch.Tensor,
    ) -> dict[str, float]:
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if not 0 <= scene_token < self.scene_vocabulary_size:
            raise ValueError("scene token is out of range")
        if program.shape != (spec.n_cells,):
            raise ValueError("hierarchical realization requires one token per block")
        if jewel_tokens.shape != (len(centers), len(ACTIVE_FACTORS)):
            raise ValueError("target Jewel tokens must align with centers")
        block_tokens = program[spec.cell_index(centers)]
        by_role = {}
        for role, name in enumerate(ACTIVE_FACTORS):
            by_role[name] = float(-self.role_log_probabilities[
                scene_token, block_tokens, role, jewel_tokens[:, role]
            ].mean())
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
        """Cast every scene-conditioned medoid block and normalize the global count."""
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if not 0 <= scene_token < self.scene_vocabulary_size:
            raise ValueError("scene token is out of range")
        if program.shape != (spec.n_cells,) or count <= 0:
            raise ValueError("hierarchical sampling requires a complete program and positive count")
        centers_out, tokens_out = [], []
        for block_id, block_token in enumerate(program.tolist()):
            pair = scene_token * self.block_vocabulary_size + block_token
            start, end = int(self.pair_offsets[pair]), int(self.pair_offsets[pair + 1])
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
            address = local.new_tensor([x, y, t])
            shape = local.new_tensor(spec.shape)
            centers_out.append(((address + (local + 1) * 0.5) / shape) * 2 - 1)
            tokens_out.append(self.jewel_tokens[start:end])
        if not centers_out:
            raise ValueError("hierarchical program contains no Jewel phrases")
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
        }


def fit_scene_block_constellation_realizer(
    fields: list[torch.Tensor],
    programs: torch.Tensor,
    scene_owners: torch.Tensor,
    *,
    block_codebook: BlockTokenCodebook,
    physical_codebook,
    likelihood_neighbors: int = 4,
    smoothing: float = 0.1,
    jitter_std: float = 0.005,
) -> tuple[SceneBlockConstellationRealizer, dict]:
    """Fit class-consistent medoids and local likelihood pools for every token pair."""
    if len(fields) != len(programs) or scene_owners.shape != (len(fields),):
        raise ValueError("fields, programs, and scene owners must align")
    semantic_scenes = int(scene_owners.max()) + 1
    if sorted(scene_owners.unique().tolist()) != list(range(semantic_scenes)):
        raise ValueError("semantic scene tokens must be dense from zero")
    if likelihood_neighbors <= 0 or smoothing <= 0 or jitter_std < 0:
        raise ValueError("hierarchical constellation settings are invalid")
    spec = GridSpec(block_codebook.block_shape, slots_per_cell=1)
    vocabulary_size = block_codebook.vocabulary_size
    descriptors, field_ids, block_ids, scene_ids = [], [], [], []
    field_locals, field_tokens, field_cells = [], [], []
    block_histograms = []
    for field_index, field in enumerate(fields):
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
        field_ids.append(torch.full(
            (spec.n_cells,), field_index, dtype=torch.long, device=field.device
        ))
        block_ids.append(torch.arange(spec.n_cells, device=field.device))
        scene_ids.append(torch.full(
            (spec.n_cells,), int(scene_owners[field_index]),
            dtype=torch.long, device=field.device,
        ))
        cells, local = block_local_coordinates(field[:, :3], spec)
        jewel = encode_active_jewel_tokens(field, physical_codebook)
        field_cells.append(cells)
        field_locals.append(local)
        field_tokens.append(jewel)
        histogram = field.new_zeros(
            spec.n_cells, len(ACTIVE_FACTORS), physical_codebook.vocabulary_size
        )
        for role in range(len(ACTIVE_FACTORS)):
            flat = cells * physical_codebook.vocabulary_size + jewel[:, role]
            histogram[:, role] = torch.bincount(
                flat,
                minlength=spec.n_cells * physical_codebook.vocabulary_size,
            ).reshape(spec.n_cells, physical_codebook.vocabulary_size)
        block_histograms.append(histogram)
    descriptors = torch.cat(descriptors)
    field_ids = torch.cat(field_ids)
    block_ids = torch.cat(block_ids)
    scene_ids = torch.cat(scene_ids)
    block_histograms = torch.stack(block_histograms)
    prototypes = block_codebook.prototypes.to(descriptors)
    scene_vocabulary_size = semantic_scenes + 1
    neighbor_fields, neighbor_blocks, neighbor_distances = [], [], []
    for scene in range(scene_vocabulary_size):
        eligible = torch.ones_like(scene_ids, dtype=torch.bool) if scene == semantic_scenes else scene_ids == scene
        eligible_descriptors = descriptors[eligible]
        eligible_fields = field_ids[eligible]
        eligible_blocks = block_ids[eligible]
        if len(eligible_descriptors) < likelihood_neighbors:
            raise ValueError("scene token has too few eligible training blocks")
        distance = torch.cdist(prototypes, eligible_descriptors).square() / descriptors.shape[1]
        values, indices = torch.topk(
            distance, likelihood_neighbors, dim=1, largest=False
        )
        neighbor_fields.append(eligible_fields[indices])
        neighbor_blocks.append(eligible_blocks[indices])
        neighbor_distances.append(values)
    neighbor_fields = torch.stack(neighbor_fields)
    neighbor_blocks = torch.stack(neighbor_blocks)
    neighbor_distances = torch.stack(neighbor_distances)
    medoid_fields = neighbor_fields[:, :, 0]
    medoid_blocks = neighbor_blocks[:, :, 0]
    medoid_distances = neighbor_distances[:, :, 0]

    pooled_histogram = fields[0].new_empty(
        scene_vocabulary_size, vocabulary_size, len(ACTIVE_FACTORS),
        physical_codebook.vocabulary_size,
    )
    for scene in range(scene_vocabulary_size):
        pooled_histogram[scene] = block_histograms[
            neighbor_fields[scene], neighbor_blocks[scene]
        ].sum(dim=1)
    probabilities = (pooled_histogram + smoothing) / (
        pooled_histogram.sum(dim=3, keepdim=True)
        + smoothing * physical_codebook.vocabulary_size
    )

    local_parts, token_parts, lengths = [], [], []
    for scene in range(scene_vocabulary_size):
        for token in range(vocabulary_size):
            field_index = int(medoid_fields[scene, token])
            block_id = int(medoid_blocks[scene, token])
            selected = field_cells[field_index] == block_id
            local_parts.append(field_locals[field_index][selected])
            token_parts.append(field_tokens[field_index][selected])
            lengths.append(int(selected.sum()))
    local_centers = torch.cat(local_parts)
    jewel_tokens = torch.cat(token_parts)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=fields[0].device)
    pair_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=fields[0].device),
        length_tensor.cumsum(0),
    ])
    realizer = SceneBlockConstellationRealizer(
        local_centers=local_centers,
        jewel_tokens=jewel_tokens,
        pair_offsets=pair_offsets,
        role_log_probabilities=probabilities.log(),
        medoid_field_indices=medoid_fields,
        medoid_block_ids=medoid_blocks,
        medoid_distances=medoid_distances,
        block_shape=spec.shape,
        semantic_scene_tokens=semantic_scenes,
        jitter_std=jitter_std,
    )
    return realizer, {
        "training_fields": len(fields),
        "semantic_scene_tokens": semantic_scenes,
        "scene_vocabulary_size_with_null": scene_vocabulary_size,
        "block_vocabulary_size": vocabulary_size,
        "stored_constellation_jewels": int(len(local_centers)),
        "empty_constellations": int((length_tensor == 0).sum()),
        "mean_constellation_jewels": float(length_tensor.float().mean()),
        "mean_correct_scene_medoid_distance": float(
            medoid_distances[:semantic_scenes].mean()
        ),
        "likelihood_neighbors": likelihood_neighbors,
        "smoothing": smoothing,
        "jitter_std": jitter_std,
    }
