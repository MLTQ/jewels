"""Select one coherent source-level Jewel program for an entire spacetime window."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.block_token_language import BlockTokenCodebook, block_descriptors
from sol.prompt_jewel_caster import ACTIVE_FACTORS, encode_active_jewel_tokens
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class CoherentSourceRealizer:
    """Emit one complete training-owned active-token field per window."""

    normalized_descriptors: torch.Tensor
    block_prototypes: torch.Tensor
    field_centers: torch.Tensor
    field_jewel_tokens: torch.Tensor
    field_block_histograms: torch.Tensor
    scene_owners: torch.Tensor
    block_shape: tuple[int, int, int]
    semantic_scene_tokens: int
    jitter_std: float
    smoothing: float

    @property
    def local_centers(self) -> torch.Tensor:
        """Expose the common storage device used by hierarchy audits."""
        return self.field_centers

    @property
    def block_vocabulary_size(self) -> int:
        return int(len(self.block_prototypes))

    @property
    def null_scene_token(self) -> int:
        return self.semantic_scene_tokens

    @property
    def scene_vocabulary_size(self) -> int:
        return self.semantic_scene_tokens + 1

    def eligible_fields(self, scene_token: int) -> torch.Tensor:
        if not 0 <= scene_token < self.scene_vocabulary_size:
            raise ValueError("scene token is out of range")
        if scene_token == self.null_scene_token:
            return torch.arange(len(self.scene_owners), device=self.scene_owners.device)
        return torch.nonzero(self.scene_owners == scene_token).flatten()

    def source_distances(
        self, scene_token: int, program: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score whole eligible fields against one addressed block program."""
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        if program.shape != (spec.n_cells,):
            raise ValueError("coherent selection requires one token per addressed block")
        eligible = self.eligible_fields(scene_token)
        prototypes = self.block_prototypes[program]
        distances = (
            self.normalized_descriptors[eligible] - prototypes[None]
        ).square().mean(dim=(1, 2))
        return eligible, distances

    def most_frequent_nonempty_token(
        self, programs: torch.Tensor, scene_token: int | None = None
    ) -> int:
        """Return the training program mode used by the pooled-null control."""
        del scene_token
        return int(torch.bincount(
            programs.flatten(), minlength=self.block_vocabulary_size
        ).argmax())

    def select_source(self, scene_token: int, program: torch.Tensor) -> tuple[int, float]:
        eligible, distances = self.source_distances(scene_token, program)
        row = int(distances.argmin())
        return int(eligible[row]), float(distances[row])

    def token_nll(
        self,
        scene_token: int,
        program: torch.Tensor,
        centers: torch.Tensor,
        jewel_tokens: torch.Tensor,
    ) -> dict[str, float]:
        """Score target active tokens under the nearest coherent source programs."""
        if jewel_tokens.shape != (len(centers), len(ACTIVE_FACTORS)):
            raise ValueError("target Jewel tokens must align with centers")
        spec = GridSpec(self.block_shape, slots_per_cell=1)
        eligible, distances = self.source_distances(scene_token, program)
        neighbors = min(4, len(eligible))
        selected = eligible[torch.topk(distances, neighbors, largest=False).indices]
        histograms = self.field_block_histograms[selected].float().sum(dim=0)
        probabilities = (histograms + self.smoothing) / (
            histograms.sum(dim=2, keepdim=True)
            + self.smoothing * histograms.shape[2]
        )
        cells = spec.cell_index(centers)
        by_role = {}
        for role, name in enumerate(ACTIVE_FACTORS):
            by_role[name] = float(-probabilities[
                cells, role, jewel_tokens[:, role]
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
        """Cast the selected source's complete quantized active-Jewel program."""
        if count <= 0:
            raise ValueError("coherent sampling requires a positive Jewel count")
        source, distance = self.select_source(scene_token, program)
        centers = self.field_centers[source]
        tokens = self.field_jewel_tokens[source]
        if len(centers) != count:
            raise ValueError("Gate 2a8 requires the frozen exact field size")
        if self.jitter_std > 0:
            centers = (
                centers
                + torch.randn(
                    centers.shape, generator=generator, device=centers.device
                )
                * self.jitter_std
            ).clamp(-0.999, 0.999)
        return centers, tokens, {
            "scene_token": scene_token,
            "selected_training_field": source,
            "mean_program_distance": distance,
            "requested_jewels": count,
            "emitted_jewels": len(centers),
            "adjustment_fraction": 0.0,
        }


def fit_coherent_source_realizer(
    fields: list[torch.Tensor],
    scene_owners: torch.Tensor,
    *,
    block_codebook: BlockTokenCodebook,
    physical_codebook,
    jitter_std: float = 0.005,
    smoothing: float = 0.1,
) -> tuple[CoherentSourceRealizer, dict]:
    """Prepare one complete active-token Jewel program per training source."""
    if not fields or scene_owners.shape != (len(fields),):
        raise ValueError("coherent fields and scene owners must align")
    if jitter_std < 0 or smoothing <= 0:
        raise ValueError("coherent realizer settings are invalid")
    expected = len(fields[0])
    if any(len(field) != expected for field in fields):
        raise ValueError("coherent realizer requires equal field sizes")
    semantic_scenes = int(scene_owners.max()) + 1
    if sorted(scene_owners.unique().tolist()) != list(range(semantic_scenes)):
        raise ValueError("semantic scene tokens must be dense from zero")
    spec = GridSpec(block_codebook.block_shape, slots_per_cell=1)
    descriptors = []
    field_tokens = []
    histograms = []
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
        tokens = encode_active_jewel_tokens(field, physical_codebook)
        field_tokens.append(tokens)
        cells = spec.cell_index(field[:, :3])
        histogram = field.new_zeros(
            spec.n_cells, len(ACTIVE_FACTORS), physical_codebook.vocabulary_size
        )
        for role in range(len(ACTIVE_FACTORS)):
            flat = cells * physical_codebook.vocabulary_size + tokens[:, role]
            histogram[:, role] = torch.bincount(
                flat,
                minlength=spec.n_cells * physical_codebook.vocabulary_size,
            ).reshape(spec.n_cells, physical_codebook.vocabulary_size)
        histograms.append(histogram.half())
    realizer = CoherentSourceRealizer(
        normalized_descriptors=torch.stack(descriptors),
        block_prototypes=block_codebook.prototypes.to(fields[0]),
        field_centers=torch.stack([field[:, :3] for field in fields]),
        field_jewel_tokens=torch.stack(field_tokens),
        field_block_histograms=torch.stack(histograms),
        scene_owners=scene_owners,
        block_shape=spec.shape,
        semantic_scene_tokens=semantic_scenes,
        jitter_std=jitter_std,
        smoothing=smoothing,
    )
    return realizer, {
        "training_fields": len(fields),
        "jewels_per_field": expected,
        "semantic_scene_tokens": semantic_scenes,
        "same_scene_candidates": {
            str(scene): int((scene_owners == scene).sum())
            for scene in range(semantic_scenes)
        },
        "one_source_choice_per_window": True,
        "likelihood_neighbors": 4,
        "smoothing": smoothing,
        "jitter_std": jitter_std,
    }
